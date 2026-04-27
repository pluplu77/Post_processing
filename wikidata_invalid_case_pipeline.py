#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# wikidata_invalid_case_pipeline_v2.py
# =============================================================================
#
# PURPOSE
# -------
# This script reads an input CSV and produces an output CSV with exactly these
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
#   formatted
#
# The script performs:
#
# 1. Entity extraction / linking for question and gold_answer
# 2. Live Wikidata existence checking for extracted entities
# 3. Property-connection checking between question-side entities and
#    gold-answer-side entities
#
#
# IMPORTANT CHANGE IN THIS VERSION
# --------------------------------
# We now use the `formatted` field as the FIRST source of entity extraction.
#
# Why?
# ----
# In your data, `formatted` often already contains the intended Wikidata entities
# explicitly as:
#   - wd:Q...
#   - entity labels with QIDs
#   - "Using entities" traces
#
# This is much more reliable than free-text NER for difficult titles like:
#   "This Is for the Lover in You"
#
# So the new extraction order is:
#
#   Question-side entities:
#     1) extract QIDs from formatted
#     2) if none found, extract explicit QIDs from question
#     3) if none found, use live Wikidata search on question text
#
#   Gold-answer-side entities:
#     1) extract explicit QIDs from gold_answer
#     2) otherwise use live Wikidata search on gold_answer
#
# NOTE:
# -----
# The formatted field may contain several QIDs. In this version, we treat those
# formatted QIDs primarily as question-side entities, because in your dataset
# formatted usually encodes the Wikidata entities used to answer the question.
#
#
# CONNECTION LOGIC
# ----------------
# We do NOT return arbitrary properties from arbitrary paths.
#
# Instead, for each pair:
#     question_entity -> ... -> gold_answer_entity
#
# we return the DISTINCT FIRST-HOP properties on the question entity that can
# reach the answer entity within MAX_HOPS.
#
#
# TAXONOMY RULES
# --------------
# - "missing node (entity)"
#     assigned when:
#       * question-side entity could not be found / linked
#       * or gold-answer-side entity could not be found / linked
#       * except for purely numeric-like gold answers
#
# - "missing edge (triplet)"
#     assigned when:
#       * both sides have valid entities
#       * but no connecting property is found
#
#
# NUMERICAL ANSWERS
# -----------------
# If gold_answer looks numeric-like, we do not force entity linking.
# In that case:
#   - gold_answer_entities stays empty
#   - no missing-node label is assigned purely because of that
#   - no edge check is attempted without answer entities
#
#
# DEPENDENCIES
# ------------
#   pip install pandas requests tqdm
#
#
# HOW TO RUN
# ----------
# 1. Change INPUT_CSV_PATH and OUTPUT_CSV_PATH below if needed.
# 2. Run:
#       python wikidata_invalid_case_pipeline_v2.py
#
# =============================================================================

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

# Input and output paths. Change if needed.
INPUT_CSV_PATH = "all_invalid_cases_first20.csv"
OUTPUT_CSV_PATH = "wikidata_pipeline_output.csv"

# Language for Wikidata label lookup.
LABEL_LANGUAGE = "en"

# Maximum hops allowed for connection search.
MAX_HOPS = 2

# HTTP config
USER_AGENT = "WikidataInvalidCasePipelineV2/1.0 (Python requests; contact: your-email@example.com)"
HEADERS_JSON = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# Wikidata endpoints
WBSEARCH_API = "https://www.wikidata.org/w/api.php"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# Timeouts / retries
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
    """Convert arbitrary value to clean string; missing -> empty string."""
    if is_missing_value(x):
        return ""
    return str(x).strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def looks_numeric_like(text: str) -> bool:
    """
    Heuristic detector for numeric / date / quantity-like answers.

    Examples considered numeric-like:
      6
      2019
      3.14
      12 km
      5 goals
      January 23, 1998
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
# LIVE HTTP / WIKIDATA HELPERS
# =============================================================================

def safe_get(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = HTTP_TIMEOUT,
    retries: int = RETRIES,
) -> requests.Response:
    """
    Robust GET helper with retries.
    """
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
    """
    Live existence check for a QID using Wikidata EntityData endpoint.
    """
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
    """
    Live Wikidata search using wbsearchentities.
    """
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
    Live label lookup for QID or PID using wbgetentities.
    Returns raw ID if label lookup fails.
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
    """Format QID as name (QID)."""
    label = get_label_for_id(qid, label_cache)
    return f"{label} ({qid})"


def format_pid_with_label(pid: str, label_cache: Dict[str, str]) -> str:
    """Format PID as property label (PID)."""
    label = get_label_for_id(pid, label_cache)
    return f"{label} ({pid})"


# =============================================================================
# ID EXTRACTION
# =============================================================================

def extract_explicit_qids(text: str) -> List[str]:
    """
    Extract explicit QIDs from text.

    Examples:
      Q567
      Angela Merkel (Q567)
      wd:Q1055
    """
    if not text:
        return []
    matches = re.findall(r"\bQ[1-9]\d*\b", text, flags=re.IGNORECASE)
    matches = [m.upper() for m in matches]
    return dedupe_preserve_order(matches)


def extract_explicit_pids(text: str) -> List[str]:
    """
    Extract explicit PIDs from text, if needed for debugging or future use.

    Examples:
      P19
      wd:P19
    """
    if not text:
        return []
    matches = re.findall(r"\bP[1-9]\d*\b", text, flags=re.IGNORECASE)
    matches = [m.upper() for m in matches]
    return dedupe_preserve_order(matches)


def extract_qids_from_formatted(formatted: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract all explicit QIDs from the formatted field, keep only those that exist.
    This is the PRIMARY extraction source in this version.
    """
    qids = extract_explicit_qids(formatted)
    qids = [qid for qid in qids if qid_exists(qid, existence_cache)]
    return dedupe_preserve_order(qids)


# =============================================================================
# HEURISTIC SURFACE EXTRACTION FOR LIVE SEARCH FALLBACK
# =============================================================================

def extract_candidate_question_surfaces(question: str) -> List[str]:
    """
    Heuristic candidate extraction from the question for fallback live search.

    This is only used if formatted gives no QIDs.

    Strategy:
      - keep the full question as a candidate
      - strip leading WH wording
      - split on common relation phrases
      - preserve longer spans
    """
    q = question.strip()
    if not q:
        return []

    q = re.sub(r"\s+", " ", q)
    candidates: List[str] = []

    # Full question sometimes helps when titles are embedded.
    candidates.append(q.strip(" ?"))

    # Remove leading WH template.
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

    # Split on some common relational patterns.
    split_phrases = [
        " of ",
        " in ",
        " on ",
        " for ",
        " from ",
        " by ",
        " where ",
        " that ",
        " which ",
        " who ",
        " whose ",
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

    Examples:
      'Ellen White, Alex Morgan, and Megan Rapinoe'
      -> ['Ellen White', 'Alex Morgan', 'Megan Rapinoe', full original string]
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
    Link a surface string to Wikidata via live search.

    Policy:
      1) if explicit QID exists in the surface, use it
      2) else live-search Wikidata
      3) take the first candidate that exists
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
# ENTITY EXTRACTION LOGIC
# =============================================================================

def extract_question_entities(question: str, formatted: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract question-side entities.

    NEW PRIORITY ORDER:
      1) extract all QIDs from formatted first
      2) if none, extract explicit QIDs from question
      3) if none, use live search on question candidate surfaces

    This is the main improvement over the previous version.
    """
    # 1) Strongest source now: formatted.
    formatted_qids = extract_qids_from_formatted(formatted, existence_cache)
    if formatted_qids:
        return formatted_qids

    # 2) Explicit QIDs inside question text.
    explicit_question_qids = [
        qid for qid in extract_explicit_qids(question)
        if qid_exists(qid, existence_cache)
    ]
    if explicit_question_qids:
        return dedupe_preserve_order(explicit_question_qids)

    # 3) Fallback to live search from question surfaces.
    out_qids: List[str] = []
    for cand in extract_candidate_question_surfaces(question):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


def extract_gold_answer_entities(gold_answer: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract gold-answer-side entities.

    Logic:
      1) if numeric-like -> return []
      2) explicit QIDs in gold_answer
      3) live search over answer candidate surfaces
    """
    gold_answer = gold_answer.strip()
    if not gold_answer:
        return []

    if looks_numeric_like(gold_answer):
        return []

    out_qids: List[str] = []

    # Explicit QIDs in answer
    for qid in extract_explicit_qids(gold_answer):
        if qid_exists(qid, existence_cache):
            out_qids.append(qid)

    # Live search fallback
    for cand in split_answer_into_candidate_surfaces(gold_answer):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


# =============================================================================
# CONNECTION CHECK
# =============================================================================

def run_sparql(query: str) -> Dict[str, Any]:
    """
    Execute a live SPARQL query against Wikidata.
    """
    resp = safe_get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"User-Agent": USER_AGENT, "Accept": "application/sparql-results+json"},
        timeout=HTTP_TIMEOUT,
        retries=RETRIES,
    )
    return resp.json()


def build_first_hop_property_query(qid1: str, qid2: str, max_hops: int) -> str:
    """
    Build a SPARQL query that returns DISTINCT first-hop properties from qid1
    that can reach qid2 within max_hops.

    Returned properties are only the FIRST edge properties on qid1.

    Example:
      qid1 --P19--> intermediate --P131--> qid2
      => return P19
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    union_blocks: List[str] = []

    # Direct 1-hop case
    union_blocks.append(f"""
    {{
      wd:{qid1} ?p1 wd:{qid2} .
      FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
    }}
    """)

    # Multi-hop outward paths
    for hops in range(2, max_hops + 1):
        lines = []
        lines.append("{")
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
        union_blocks.append("\n".join(lines))

    query = "SELECT DISTINCT ?p1 WHERE {\n"
    query += "\nUNION\n".join(union_blocks)
    query += "\n}\nORDER BY ?p1"
    return query


def find_connecting_first_hop_pids(qid1: str, qid2: str, max_hops: int) -> List[str]:
    """
    Return all distinct first-hop PIDs on qid1 that can reach qid2.
    """
    query = build_first_hop_property_query(qid1, qid2, max_hops)
    data = run_sparql(query)
    bindings = data.get("results", {}).get("bindings", [])

    pids: List[str] = []
    for row in bindings:
        prop_uri = row["p1"]["value"]
        pid = prop_uri.rsplit("/", 1)[-1]
        if re.fullmatch(r"P[1-9]\d*", pid):
            pids.append(pid)

    return dedupe_preserve_order(pids)


# =============================================================================
# ROW PROCESSING
# =============================================================================

def should_skip_row(question: str, gold_answer: str, formatted: str) -> bool:
    """
    Skip rows that look like non-QA / empty rows.
    """
    if not question and not gold_answer and not formatted:
        return True
    return False


def process_row(
    row: pd.Series,
    existence_cache: Dict[str, bool],
    label_cache: Dict[str, str],
) -> Dict[str, Any]:
    """
    Process one row into the required output format.
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
        "formatted": formatted,
    }

    if should_skip_row(question, gold_answer, formatted):
        return out

    # -----------------------------------------------------------------
    # Step 1: Extract entities
    # -----------------------------------------------------------------
    question_qids = extract_question_entities(question, formatted, existence_cache)
    answer_qids = extract_gold_answer_entities(gold_answer, existence_cache)

    # Format entity lists as name (QID)
    out["question_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in question_qids
    )
    out["gold_answer_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in answer_qids
    )

    # -----------------------------------------------------------------
    # Step 2: Missing-node logic
    # -----------------------------------------------------------------
    answer_is_numeric_like = looks_numeric_like(gold_answer)

    missing_question_entity = (question != "" and len(question_qids) == 0)
    missing_answer_entity = (gold_answer != "" and not answer_is_numeric_like and len(answer_qids) == 0)

    if missing_question_entity or missing_answer_entity:
        out["inconsistency_taxonomy"] = "missing node (entity)"
        return out

    # -----------------------------------------------------------------
    # Step 3: Numeric-like answers are not forced into edge checking
    # -----------------------------------------------------------------
    if answer_is_numeric_like and len(answer_qids) == 0:
        return out

    # -----------------------------------------------------------------
    # Step 4: Check connections between all question × answer pairs
    # -----------------------------------------------------------------
    aggregated_pids: List[str] = []

    for q_qid in question_qids:
        for a_qid in answer_qids:
            try:
                pids = find_connecting_first_hop_pids(q_qid, a_qid, MAX_HOPS)
                aggregated_pids.extend(pids)
            except Exception:
                continue

    aggregated_pids = dedupe_preserve_order(aggregated_pids)

    if not aggregated_pids:
        out["inconsistency_taxonomy"] = "missing edge (triplet)"
        return out

    out["property_number"] = len(aggregated_pids)
    out["property_name"] = ";".join(
        format_pid_with_label(pid, label_cache) for pid in aggregated_pids
    )

    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    """
    Main pipeline:
      1) read input CSV
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
        "formatted",
    ]
    out_df = out_df[final_columns]

    print(f"Writing output CSV: {OUTPUT_CSV_PATH}")
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")
    print(f"Output saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()