#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# wikidata_invalid_case_pipeline_v3.py
# =============================================================================
#
# PURPOSE
# -------
# This script reads an input CSV and outputs a refined CSV with exactly these
# columns:
#
#   question
#   gold_answer
#   answer
#   inconsistency_taxonomy
#   question_entities
#   gold_answer_entities
#   property_number
#   property_name
#   connection_path
#   formatted
#
# MAIN CAPABILITIES
# -----------------
# 1. Extract question-side entities, primarily from `formatted`
# 2. Extract gold-answer-side entities from `gold_answer`
# 3. Verify QID existence live in Wikidata
# 4. Check whether question entities can connect to answer entities
# 5. Return:
#      - first-hop property count
#      - first-hop property names
#      - one or more actual graph paths
#
#
# IMPORTANT CHANGE IN THIS VERSION
# --------------------------------
# Previous versions only returned the first-hop property/properties on the
# question entity that can eventually reach the answer entity.
#
# This version also returns the ACTUAL PATH, for example:
#
#   This Is for the Lover in You (Q7786031)
#   -> followed by (P156)
#   -> Every Time I Close My Eyes (Q5417613)
#   -> performer (P175)
#   -> Babyface (Q344983)
#
#
# PATH INTERPRETATION
# -------------------
# For each pair:
#     question_entity -> ... -> gold_answer_entity
#
# we search outward paths up to MAX_HOPS and return:
#
#   - the first-hop property p1
#   - the full path edges [(node1, pid1, node2), (node2, pid2, node3), ...]
#
# The `property_name` column still contains DISTINCT FIRST-HOP properties,
# because that matches your prior output design.
#
# The new `connection_path` column contains the actual concrete path(s).
#
#
# TAXONOMY RULES
# --------------
# - "missing node (entity)"
#     if required entities cannot be linked / do not exist
#
# - "missing edge (triplet)"
#     if both sides have entities but no connecting path is found
#
#
# DEPENDENCIES
# ------------
#   pip install pandas requests tqdm
#
#
# HOW TO RUN
# ----------
# 1. Edit INPUT_CSV_PATH and OUTPUT_CSV_PATH if needed.
# 2. Run:
#       python wikidata_invalid_case_pipeline_v3.py
#
# =============================================================================

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_CSV_PATH = "/mnt/data/all_invalid_cases.csv"
OUTPUT_CSV_PATH = "/mnt/data/wikidata_pipeline_output_v3.csv"

LABEL_LANGUAGE = "en"
MAX_HOPS = 2

USER_AGENT = "WikidataInvalidCasePipelineV3/1.0 (Python requests; contact: your-email@example.com)"
HEADERS_JSON = {"User-Agent": USER_AGENT, "Accept": "application/json"}

WBSEARCH_API = "https://www.wikidata.org/w/api.php"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

HTTP_TIMEOUT = 25
RETRIES = 2


# =============================================================================
# BASIC HELPERS
# =============================================================================

def is_missing_value(x: Any) -> bool:
    """Return True if a value is effectively missing."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if str(x).strip() == "":
        return True
    return False


def normalize_text(x: Any) -> str:
    """Convert arbitrary value to a clean string."""
    if is_missing_value(x):
        return ""
    return str(x).strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def looks_numeric_like(text: str) -> bool:
    """
    Heuristic detector for numeric/date/quantity-like answers.

    These are not forced into entity-linking failure.
    """
    t = text.strip().lower()
    if not t:
        return False

    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?", t):
        return True

    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?\s+[a-z%°/.-]+", t):
        return True

    if re.search(r"\b\d{4}\b", t) and re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        t
    ):
        return True

    return False


# =============================================================================
# HTTP / WIKIDATA HELPERS
# =============================================================================

def safe_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT,
    retries: int = RETRIES,
) -> requests.Response:
    """HTTP GET helper with retry logic."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(1.2 * (attempt + 1))
            else:
                raise RuntimeError(f"HTTP request failed: {exc}") from exc
    raise RuntimeError(f"HTTP request failed: {last_exc}")


def qid_exists(qid: str, existence_cache: Dict[str, bool]) -> bool:
    """Live existence check for a QID."""
    if qid in existence_cache:
        return existence_cache[qid]

    url = ENTITY_DATA_URL.format(qid)
    try:
        resp = safe_get(url, headers={"User-Agent": USER_AGENT})
        exists = (resp.status_code == 200)
    except Exception:
        exists = False

    existence_cache[qid] = exists
    return exists


def wbsearch_entities(search_text: str, limit: int = 5, language: str = LABEL_LANGUAGE) -> List[Dict[str, Any]]:
    """Live Wikidata search using wbsearchentities."""
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": language,
        "uselang": language,
        "search": search_text,
        "limit": limit,
        "type": "item",
    }
    resp = safe_get(WBSEARCH_API, params=params, headers=HEADERS_JSON)
    data = resp.json()
    return data.get("search", [])


def get_label_for_id(entity_id: str, label_cache: Dict[str, str], language: str = LABEL_LANGUAGE) -> str:
    """
    Get human-readable label for QID or PID.
    Returns raw ID if lookup fails.
    """
    if entity_id in label_cache:
        return label_cache[entity_id]

    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels",
        "languages": language,
        "format": "json",
    }

    try:
        resp = safe_get(WBSEARCH_API, params=params, headers={"User-Agent": USER_AGENT})
        data = resp.json()
        label = data["entities"][entity_id]["labels"][language]["value"]
    except Exception:
        label = entity_id

    label_cache[entity_id] = label
    return label


def format_qid_with_label(qid: str, label_cache: Dict[str, str]) -> str:
    """Format entity as name (QID)."""
    return f"{get_label_for_id(qid, label_cache)} ({qid})"


def format_pid_with_label(pid: str, label_cache: Dict[str, str]) -> str:
    """Format property as label (PID)."""
    return f"{get_label_for_id(pid, label_cache)} ({pid})"


# =============================================================================
# ID EXTRACTION
# =============================================================================

def extract_explicit_qids(text: str) -> List[str]:
    """Extract explicit QIDs from text."""
    if not text:
        return []
    matches = re.findall(r"\bQ[1-9]\d*\b", text, flags=re.IGNORECASE)
    return dedupe_preserve_order([m.upper() for m in matches])


def extract_qids_from_formatted(formatted: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract QIDs from formatted first.

    This is our primary question-side source because formatted often already
    encodes the intended Wikidata entities.
    """
    qids = extract_explicit_qids(formatted)
    qids = [qid for qid in qids if qid_exists(qid, existence_cache)]
    return dedupe_preserve_order(qids)


# =============================================================================
# LIVE SEARCH FALLBACK
# =============================================================================

def extract_candidate_question_surfaces(question: str) -> List[str]:
    """
    Heuristic question-surface extraction for live search fallback.
    """
    q = question.strip()
    if not q:
        return []

    q = re.sub(r"\s+", " ", q)
    candidates: List[str] = [q.strip(" ?")]

    q_lower = q.lower()
    prefixes = [
        r"^who\s+",
        r"^what\s+",
        r"^where\s+",
        r"^when\s+",
        r"^which\s+",
        r"^how many\s+",
        r"^how much\s+",
        r"^how old\s+",
        r"^name\s+the\s+",
        r"^tell me\s+",
    ]

    stripped = q
    for p in prefixes:
        m = re.match(p, q_lower)
        if m:
            stripped = q[len(m.group(0)):]
            break

    stripped = stripped.strip(" ?")
    if stripped:
        candidates.append(stripped)

    split_phrases = [
        " of ", " in ", " on ", " for ", " from ", " by ",
        " where ", " that ", " which ", " who ", " whose "
    ]

    pieces = [stripped]
    for phrase in split_phrases:
        new_pieces = []
        for piece in pieces:
            new_pieces.extend(piece.split(phrase))
        pieces = new_pieces

    for piece in pieces:
        piece = piece.strip(" ,.?;:()[]{}")
        if len(piece) >= 3:
            candidates.append(piece)

    return dedupe_preserve_order(candidates)


def split_answer_into_candidate_surfaces(gold_answer: str) -> List[str]:
    """
    Split gold_answer into candidate entity surfaces.
    """
    t = gold_answer.strip()
    if not t:
        return []

    t_norm = re.sub(r"\s+and\s+", ", ", t, flags=re.IGNORECASE)
    t_norm = re.sub(r"\s*&\s*", ", ", t_norm)

    parts = [p.strip() for p in t_norm.split(",")]
    parts = [p for p in parts if p]

    if len(parts) > 1:
        return dedupe_preserve_order(parts + [t])
    return parts


def pick_best_qid_for_surface(surface: str, existence_cache: Dict[str, bool]) -> Optional[str]:
    """
    Link a free-text surface to Wikidata using live search.
    """
    surface = surface.strip()
    if not surface:
        return None

    explicit_qids = extract_explicit_qids(surface)
    for qid in explicit_qids:
        if qid_exists(qid, existence_cache):
            return qid

    try:
        candidates = wbsearch_entities(surface, limit=5)
    except Exception:
        return None

    for cand in candidates:
        qid = cand.get("id")
        if qid and re.fullmatch(r"Q[1-9]\d*", qid) and qid_exists(qid, existence_cache):
            return qid

    return None


# =============================================================================
# ENTITY EXTRACTION
# =============================================================================

def extract_question_entities(question: str, formatted: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Question-side extraction order:
      1) formatted QIDs first
      2) explicit QIDs in question
      3) live search fallback
    """
    formatted_qids = extract_qids_from_formatted(formatted, existence_cache)
    if formatted_qids:
        return formatted_qids

    explicit_question_qids = [
        qid for qid in extract_explicit_qids(question)
        if qid_exists(qid, existence_cache)
    ]
    if explicit_question_qids:
        return dedupe_preserve_order(explicit_question_qids)

    out_qids: List[str] = []
    for cand in extract_candidate_question_surfaces(question):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


def extract_gold_answer_entities(gold_answer: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Gold-answer extraction:
      1) skip if numeric-like
      2) explicit QIDs
      3) live search fallback
    """
    gold_answer = gold_answer.strip()
    if not gold_answer:
        return []

    if looks_numeric_like(gold_answer):
        return []

    out_qids: List[str] = []

    for qid in extract_explicit_qids(gold_answer):
        if qid_exists(qid, existence_cache):
            out_qids.append(qid)

    for cand in split_answer_into_candidate_surfaces(gold_answer):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


# =============================================================================
# PATH SEARCH
# =============================================================================

def run_sparql(query: str) -> Dict[str, Any]:
    """Execute a live SPARQL query."""
    resp = safe_get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=HTTP_TIMEOUT,
        retries=RETRIES,
    )
    return resp.json()


def build_exact_path_query(qid1: str, qid2: str, hops: int) -> str:
    """
    Build a SPARQL query that searches for ONE outward path of exactly `hops`
    edges from qid1 to qid2.

    Example for hops=2:
      wd:Q1 ?p1 ?n1 .
      ?n1 ?p2 wd:Q2 .

    Returns:
      variables ?p1, ?n1, ?p2
    """
    if hops < 1:
        raise ValueError("hops must be >= 1")

    lines = ["SELECT * WHERE {"]

    if hops == 1:
        lines.append(f"  wd:{qid1} ?p1 wd:{qid2} .")
        lines.append('  FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))')
        lines.append("}")
        lines.append("LIMIT 1")
        return "\n".join(lines)

    lines.append(f"  wd:{qid1} ?p1 ?n1 .")
    lines.append('  FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))')
    lines.append('  FILTER(STRSTARTS(STR(?n1), "http://www.wikidata.org/entity/Q"))')

    for i in range(1, hops - 1):
        lines.append(f"  ?n{i} ?p{i+1} ?n{i+1} .")
        lines.append(f'  FILTER(STRSTARTS(STR(?p{i+1}), "http://www.wikidata.org/prop/direct/"))')
        lines.append(f'  FILTER(STRSTARTS(STR(?n{i+1}), "http://www.wikidata.org/entity/Q"))')

    lines.append(f"  ?n{hops-1} ?p{hops} wd:{qid2} .")
    lines.append(f'  FILTER(STRSTARTS(STR(?p{hops}), "http://www.wikidata.org/prop/direct/"))')

    for i in range(1, hops):
        for j in range(i + 1, hops):
            lines.append(f"  FILTER(?n{i} != ?n{j})")

    lines.append("}")
    lines.append("LIMIT 1")
    return "\n".join(lines)


def parse_path_from_binding(row: Dict[str, Any], qid1: str, qid2: str, hops: int) -> List[Tuple[str, str, str]]:
    """
    Convert one SPARQL result row into a path represented as:
      [(source_qid, pid, target_qid), ...]

    Example:
      [("Q7786031", "P156", "Q5417613"),
       ("Q5417613", "P175", "Q344983")]
    """
    path_edges: List[Tuple[str, str, str]] = []

    if hops == 1:
        pid = row["p1"]["value"].rsplit("/", 1)[-1]
        path_edges.append((qid1, pid, qid2))
        return path_edges

    current_source = qid1
    for i in range(1, hops + 1):
        pid = row[f"p{i}"]["value"].rsplit("/", 1)[-1]
        if i < hops:
            next_qid = row[f"n{i}"]["value"].rsplit("/", 1)[-1]
        else:
            next_qid = qid2

        path_edges.append((current_source, pid, next_qid))
        current_source = next_qid

    return path_edges


def find_one_outward_path(qid1: str, qid2: str, max_hops: int) -> Optional[List[Tuple[str, str, str]]]:
    """
    Find one outward path from qid1 to qid2 up to max_hops.

    We try:
      1 hop, then 2 hops, then 3 hops, ...

    The first found path is returned.
    """
    for hops in range(1, max_hops + 1):
        query = build_exact_path_query(qid1, qid2, hops)
        data = run_sparql(query)
        bindings = data.get("results", {}).get("bindings", [])
        if bindings:
            return parse_path_from_binding(bindings[0], qid1, qid2, hops)
    return None


def path_to_readable_string(path_edges: List[Tuple[str, str, str]], label_cache: Dict[str, str]) -> str:
    """
    Convert a path edge list into a readable string like:

      This Is for the Lover in You (Q7786031)
      -> followed by (P156)
      -> Every Time I Close My Eyes (Q5417613)
      -> performer (P175)
      -> Babyface (Q344983)
    """
    if not path_edges:
        return ""

    parts = [format_qid_with_label(path_edges[0][0], label_cache)]
    for source_qid, pid, target_qid in path_edges:
        parts.append(format_pid_with_label(pid, label_cache))
        parts.append(format_qid_with_label(target_qid, label_cache))

    return " -> ".join(parts)


def get_first_hop_pids_from_path(path_edges: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract the first-hop PID from a found path.
    """
    if not path_edges:
        return []
    return [path_edges[0][1]]


# =============================================================================
# ROW PROCESSING
# =============================================================================

def should_skip_row(question: str, gold_answer: str, formatted: str) -> bool:
    """Skip rows that are effectively empty / non-QA."""
    return (not question and not gold_answer and not formatted)


def process_row(
    row: pd.Series,
    existence_cache: Dict[str, bool],
    label_cache: Dict[str, str],
) -> Dict[str, Any]:
    """
    Process one row into the final output format.
    """
    question = normalize_text(row.get("question"))
    gold_answer = normalize_text(row.get("gold_answer"))
    answer = normalize_text(row.get("answer"))
    formatted = normalize_text(row.get("formatted"))

    out = {
        "question": question,
        "gold_answer": gold_answer,
        "answer": answer,
        "inconsistency_taxonomy": "",
        "question_entities": "",
        "gold_answer_entities": "",
        "property_number": "",
        "property_name": "",
        "connection_path": "",
        "formatted": formatted,
    }

    if should_skip_row(question, gold_answer, formatted):
        return out

    # -------------------------------------------------------------
    # Step 1: Extract entities
    # -------------------------------------------------------------
    question_qids = extract_question_entities(question, formatted, existence_cache)
    answer_qids = extract_gold_answer_entities(gold_answer, existence_cache)

    out["question_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in question_qids
    )
    out["gold_answer_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in answer_qids
    )

    # -------------------------------------------------------------
    # Step 2: Missing node logic
    # -------------------------------------------------------------
    answer_is_numeric_like = looks_numeric_like(gold_answer)

    missing_question_entity = (question != "" and len(question_qids) == 0)
    missing_answer_entity = (gold_answer != "" and not answer_is_numeric_like and len(answer_qids) == 0)

    if missing_question_entity or missing_answer_entity:
        out["inconsistency_taxonomy"] = "missing node (entity)"
        return out

    # Numeric-like answers are not forced into edge checking.
    if answer_is_numeric_like and len(answer_qids) == 0:
        return out

    # -------------------------------------------------------------
    # Step 3: Find real paths for all question × answer pairs
    # -------------------------------------------------------------
    all_first_hop_pids: List[str] = []
    all_readable_paths: List[str] = []

    for q_qid in question_qids:
        for a_qid in answer_qids:
            try:
                path_edges = find_one_outward_path(q_qid, a_qid, MAX_HOPS)
            except Exception:
                path_edges = None

            if path_edges:
                all_first_hop_pids.extend(get_first_hop_pids_from_path(path_edges))
                all_readable_paths.append(path_to_readable_string(path_edges, label_cache))

    all_first_hop_pids = dedupe_preserve_order(all_first_hop_pids)
    all_readable_paths = dedupe_preserve_order(all_readable_paths)

    # If no path exists, label as missing edge.
    if not all_readable_paths:
        out["inconsistency_taxonomy"] = "missing edge (triplet)"
        return out

    out["property_number"] = len(all_first_hop_pids)
    out["property_name"] = ";".join(
        format_pid_with_label(pid, label_cache) for pid in all_first_hop_pids
    )
    out["connection_path"] = " || ".join(all_readable_paths)

    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main pipeline:
      1) load CSV
      2) process rows with tqdm
      3) save output CSV
    """
    print(f"Reading input CSV: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    existence_cache: Dict[str, bool] = {}
    label_cache: Dict[str, str] = {}
    output_records: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="processing rows"):
        output_records.append(process_row(row, existence_cache, label_cache))

    out_df = pd.DataFrame(output_records)

    final_columns = [
        "question",
        "gold_answer",
        "answer",
        "inconsistency_taxonomy",
        "question_entities",
        "gold_answer_entities",
        "property_number",
        "property_name",
        "connection_path",
        "formatted",
    ]
    out_df = out_df[final_columns]

    print(f"Writing output CSV: {OUTPUT_CSV_PATH}")
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")
    print(f"Output saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
