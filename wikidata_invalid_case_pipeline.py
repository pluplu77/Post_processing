#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# wikidata_invalid_case_pipeline.py
# =============================================================================
#
# PURPOSE
# -------
# This script reads an input CSV of invalid QA cases and builds a new CSV that:
#
# 1. Only keeps these input columns:
#       file_path, question, gold_answer, answer, formatted
#
# 2. Adds:
#       inconsistency_taxonomy
#       question_entities
#       gold_answer_entities
#       property_number
#       property_name
#
# 3. Uses LIVE Wikidata requests to:
#       - detect / link entities from question and gold_answer
#       - verify that linked entities actually exist
#       - check whether question-side entities connect to gold_answer-side entities
#         in Wikidata through a bounded "local statement graph" search
#
#
# TASK LOGIC
# ----------
# A. ENTITY EXTRACTION / LINKING
#    We try to obtain Wikidata QIDs for:
#       - topic entities in the question
#       - answer entities in gold_answer
#
#    The pipeline uses a practical rule-based + Wikidata-search approach:
#
#    1) If text already contains an explicit QID, such as:
#           Q567
#           Angela Merkel (Q567)
#       we use that directly.
#
#    2) Otherwise, we try to extract candidate surface strings from the question
#       or gold_answer and search Wikidata live using the wbsearchentities API.
#
#    3) We also inspect the 'formatted' column for explicit wd:Q... entity mentions.
#       This can help because your formatted field often contains:
#           "Using entities:"
#           "- Abbey Road (wd:Q173643)"
#
#
# B. EXISTENCE CHECK
#    After we obtain a QID candidate, we verify it exists using:
#       https://www.wikidata.org/wiki/Special:EntityData/QID.json
#
#    If a required entity cannot be linked or does not exist:
#       inconsistency_taxonomy = "missing node (entity)"
#
#
# C. CONNECTION CHECK
#    If valid entities exist on both sides, we check whether any question entity
#    can connect to any gold_answer entity.
#
#    IMPORTANT INTERPRETATION
#    ------------------------
#    We do NOT return arbitrary properties from arbitrary graph paths.
#
#    Instead, for each pair:
#       question_entity -> ... -> gold_answer_entity
#
#    we return the DISTINCT FIRST-HOP properties on the question entity that can
#    reach the gold answer within MAX_HOPS.
#
#    Example:
#       Angela Merkel (Q567) -- place of birth (P19) --> Eimsbüttel (...)
#       Eimsbüttel (...)     -- located in the administrative territorial entity (P131) --> Hamburg (Q1055)
#
#    Output property:
#       place of birth (P19)
#
#
# D. MULTIPLE ANSWERS / MULTIPLE ENTITIES
#    If multiple answer entities are extracted, we test all combinations:
#       each question entity  ×  each answer entity
#
#    We then aggregate and deduplicate all matched first-hop properties.
#
#f
# E. NUMERICAL ANSWERS
#    If gold_answer is purely numeric / quantity / date-like and no entity can
#    reasonably be linked, we do NOT force entity linking.
#
#    In that case:
#       - we leave entity/property columns empty
#       - we do NOT automatically label it as "missing node (entity)"
#
#    This is important because many numeric answers are valid non-entity answers.
#
#
# OUTPUT COLUMNS
# --------------
# The output CSV columns are:
#
#   file_path
#   question
#   gold_answer
#   answer
#   formatted
#   inconsistency_taxonomy
#   question_entities
#   gold_answer_entities
#   property_number
#   property_name
#
# where:
#   question_entities   = semicolon-separated "name (QID)"
#   gold_answer_entities = semicolon-separated "name (QID)"
#   property_name       = semicolon-separated "property label (PID)"
#
#
# DEPENDENCIES
# ------------
#   pip install pandas requests tqdm
#
#
# HOW TO RUN
# ----------
# 1. Change INPUT_CSV_PATH and OUTPUT_CSV_PATH below.
# 2. Run:
#       python wikidata_invalid_case_pipeline.py
#
# =============================================================================

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm


# =============================================================================
# CONFIGURATION
# =============================================================================

# -------------------------------------------------------------------------
# Hardcoded input and output paths.
# Replace these with your real local paths if needed.
# -------------------------------------------------------------------------
INPUT_CSV_PATH = "all_invalid_cases_first20.csv"
OUTPUT_CSV_PATH = "wikidata_pipeline_output.csv"

# -------------------------------------------------------------------------
# Language used when requesting labels from Wikidata.
# -------------------------------------------------------------------------
LABEL_LANGUAGE = "en"

# -------------------------------------------------------------------------
# Maximum number of hops allowed when checking whether a question entity can
# connect to a gold_answer entity.
#
# MAX_HOPS = 1:
#   Q1 --P?--> Q2
#
# MAX_HOPS = 2:
#   Q1 --P1--> intermediate --P2--> Q2
#
# MAX_HOPS = 3:
#   Q1 --P1--> n1 --P2--> n2 --P3--> Q2
#
# 2 is usually a good starting point for your use case.
# -------------------------------------------------------------------------
MAX_HOPS = 2

# -------------------------------------------------------------------------
# HTTP headers. Wikidata recommends using a descriptive User-Agent.
# -------------------------------------------------------------------------
USER_AGENT = "WikidataInvalidCasePipeline/1.0 (Python requests; contact: your-email@example.com)"
HEADERS_JSON = {"User-Agent": USER_AGENT, "Accept": "application/json"}

# -------------------------------------------------------------------------
# Endpoints used in this script.
# -------------------------------------------------------------------------
WBSEARCH_API = "https://www.wikidata.org/w/api.php"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

# -------------------------------------------------------------------------
# Timeout / retry settings for live Wikidata checks.
# -------------------------------------------------------------------------
HTTP_TIMEOUT = 25
RETRIES = 2


# =============================================================================
# SMALL TEXT / TYPE HELPERS
# =============================================================================

def is_missing_value(x: Any) -> bool:
    """
    Return True if a value is effectively missing.

    This is needed because the CSV contains NaN values in some rows,
    especially timing summary rows.
    """
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if str(x).strip() == "":
        return True
    return False


def normalize_text(x: Any) -> str:
    """
    Convert arbitrary value to a clean string.
    Missing values become an empty string.
    """
    if is_missing_value(x):
        return ""
    return str(x).strip()


def looks_numeric_like(text: str) -> bool:
    """
    Heuristic detector for non-entity answers such as:
      - 6
      - 3.14
      - 2019
      - 12 km
      - 5 goals
      - 23 January 1998

    We use this to avoid falsely labeling purely numeric answers as
    "missing node (entity)".

    This is intentionally heuristic, not perfect.
    """
    t = text.strip().lower()
    if not t:
        return False

    # Pure number / decimal / commas
    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?", t):
        return True

    # Number followed by short unit / token
    if re.fullmatch(r"[-+]?\d[\d,]*(\.\d+)?\s+[a-z%°/.-]+", t):
        return True

    # Date-like strings (simple heuristic)
    if re.search(r"\b\d{4}\b", t) and re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b", t):
        return True

    return False


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """
    Remove duplicates from a list while preserving first-seen order.
    """
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# =============================================================================
# WIKIDATA API HELPERS
# =============================================================================

def safe_get(url: str, *, params: Optional[Dict[str, Any]] = None,
             headers: Optional[Dict[str, str]] = None,
             timeout: int = HTTP_TIMEOUT,
             retries: int = RETRIES) -> requests.Response:
    """
    Robust GET helper with simple retry logic.

    We use this for all live Wikidata checks.
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
    Check whether a QID exists in Wikidata using the live EntityData endpoint.

    Result is cached because the same QIDs may occur many times.
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
    Search Wikidata entities live using wbsearchentities.

    Returns a list of candidate items/properties depending on the 'type' parameter.
    Here we use the default item search.

    This is the main live linking step when no explicit QID is available.
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
    Get a human-readable label for a QID or PID using wbgetentities.

    Returns the raw ID if no label is found.

    Examples:
      Q567  -> Angela Merkel
      Q1055 -> Hamburg
      P19   -> place of birth
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
    """
    Convert:
      Q567
    into:
      Angela Merkel (Q567)
    """
    label = get_label_for_id(qid, label_cache)
    return f"{label} ({qid})"


def format_pid_with_label(pid: str, label_cache: Dict[str, str]) -> str:
    """
    Convert:
      P19
    into:
      place of birth (P19)
    """
    label = get_label_for_id(pid, label_cache)
    return f"{label} ({pid})"


# =============================================================================
# EXPLICIT ID EXTRACTION FROM TEXT
# =============================================================================

def extract_explicit_qids(text: str) -> List[str]:
    """
    Extract explicit QIDs from arbitrary text.

    Examples:
      "Angela Merkel (Q567)" -> ["Q567"]
      "wd:Q1055"             -> ["Q1055"]
      "Q42"                  -> ["Q42"]

    We preserve order and deduplicate.
    """
    if not text:
        return []
    matches = re.findall(r"\bQ[1-9]\d*\b", text, flags=re.IGNORECASE)
    matches = [m.upper() for m in matches]
    return dedupe_preserve_order(matches)


def extract_qids_from_formatted(formatted: str) -> List[str]:
    """
    Extract explicit QIDs from the formatted field.

    Your 'formatted' column often includes:
      - wd:Q...
      - 'Using entities:'
      - human-readable entity descriptions

    We use these QIDs as an auxiliary source for entity linking.
    """
    return extract_explicit_qids(formatted)


# =============================================================================
# QUESTION / ANSWER ENTITY EXTRACTION
# =============================================================================

def extract_candidate_quoted_or_titled_spans(question: str) -> List[str]:
    """
    Heuristic extraction of candidate entity strings from the question.

    Why heuristic?
    --------------
    We are not using a heavy NER model here because:
      - your data is tightly coupled to Wikidata linking
      - live Wikidata search is more useful than generic NER labels
      - rule-based extraction is easier to debug for this pipeline

    Strategy:
      1) remove leading WH / auxiliary wording
      2) split on punctuation and some relation phrases
      3) keep noun-like tail spans as candidate strings

    This is intentionally simple and can be refined later.
    """
    q = question.strip()
    if not q:
        return []

    # Normalize whitespace
    q = re.sub(r"\s+", " ", q)

    candidates = []

    # If question contains quoted text, quoted strings are good candidates.
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', q)
    for tup in quoted:
        for x in tup:
            x = x.strip()
            if x:
                candidates.append(x)

    # Remove leading question templates
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
    q2 = q.lower()
    cut = question.strip()
    for p in prefixes:
        m = re.match(p, q2)
        if m:
            cut = q[len(m.group(0)):]
            break

    # Split on common relation phrases. The right-side chunks often contain the topic entity.
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

    pieces = [cut]
    for phrase in split_phrases:
        new_pieces = []
        for piece in pieces:
            new_pieces.extend(piece.split(phrase))
        pieces = new_pieces

    # Keep moderately informative pieces.
    for piece in pieces:
        piece = piece.strip(" ,.?;:()[]{}")
        if len(piece) >= 3:
            candidates.append(piece)

    return dedupe_preserve_order(candidates)


def split_answer_into_candidate_entities(answer_text: str) -> List[str]:
    """
    Split gold_answer into candidate entity strings.

    Examples:
      "Angela Merkel" -> ["Angela Merkel"]
      "Ellen White, Alex Morgan, and Megan Rapinoe" ->
          ["Ellen White", "Alex Morgan", "Megan Rapinoe"]

    This is a heuristic splitter; it can be improved later.
    """
    t = answer_text.strip()
    if not t:
        return []

    # Normalize conjunctions to comma for easier splitting.
    t = re.sub(r"\s+and\s+", ", ", t, flags=re.IGNORECASE)
    t = re.sub(r"\s*&\s*", ", ", t)

    parts = [p.strip() for p in t.split(",")]
    parts = [p for p in parts if p]

    # If splitting made things worse, keep the original too.
    if len(parts) > 1:
        return dedupe_preserve_order(parts + [answer_text.strip()])
    return parts


def pick_best_qid_for_surface(surface: str,
                              existence_cache: Dict[str, bool]) -> Optional[str]:
    """
    Link a surface string to a Wikidata QID using live search.

    Current policy:
      - query wbsearchentities
      - keep the first existing QID candidate

    This is a simple baseline and intentionally conservative.
    It can be improved later with better ranking logic.
    """
    surface = surface.strip()
    if not surface:
        return None

    # 1) If explicit QID already appears in the text, trust that first.
    explicit_qids = extract_explicit_qids(surface)
    for qid in explicit_qids:
        if qid_exists(qid, existence_cache):
            return qid

    # 2) Live search on Wikidata.
    try:
        candidates = wbsearch_entities(surface, limit=5)
    except Exception:
        return None

    for cand in candidates:
        qid = cand.get("id")
        if qid and re.fullmatch(r"Q[1-9]\d*", qid) and qid_exists(qid, existence_cache):
            return qid

    return None


def extract_question_entities(question: str,
                              formatted: str,
                              existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract / link question-side entities.

    Order of evidence:
      1) explicit QIDs inside the question
      2) Wikidata search over question candidate spans
      3) fallback to QIDs mentioned in formatted

    Why not rely only on 'formatted'?
    ---------------------------------
    Because formatted may contain many unrelated QIDs from a failed SPARQL attempt.
    We use it only as a fallback or weak cue.
    """
    out_qids: List[str] = []

    # Strongest signal: explicit QIDs in question.
    for qid in extract_explicit_qids(question):
        if qid_exists(qid, existence_cache):
            out_qids.append(qid)

    # Heuristic surface extraction from question text.
    question_candidates = extract_candidate_quoted_or_titled_spans(question)
    for cand in question_candidates:
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    # Fallback to formatted QIDs only if we found nothing from the question itself.
    if not out_qids:
        for qid in extract_qids_from_formatted(formatted):
            if qid_exists(qid, existence_cache):
                out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


def extract_gold_answer_entities(gold_answer: str,
                                 existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract / link gold-answer-side entities.

    If the answer looks numeric-like, we skip entity linking and return [].
    That avoids false 'missing node (entity)' labels for pure numeric answers.
    """
    gold_answer = gold_answer.strip()
    if not gold_answer:
        return []

    if looks_numeric_like(gold_answer):
        return []

    out_qids: List[str] = []

    # Explicit QIDs first.
    for qid in extract_explicit_qids(gold_answer):
        if qid_exists(qid, existence_cache):
            out_qids.append(qid)

    # Otherwise search over answer splits.
    for cand in split_answer_into_candidate_entities(gold_answer):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid:
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


# =============================================================================
# CONNECTION CHECK VIA SPARQL
# =============================================================================

def run_sparql(query: str) -> Dict[str, Any]:
    """
    Execute a SPARQL query against Wikidata live.
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

    Key design:
      - the FIRST edge is always outgoing from qid1
      - therefore returned properties are properties of qid1
      - only direct truthy entity-to-entity edges are traversed

    Example with 2 hops:
      wd:Q567 ?p1 ?n1 .
      ?n1 ?p2 wd:Q1055 .
      return ?p1
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    union_blocks = []

    # 1-hop case: qid1 -> qid2 directly
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

        # Intermediate outward edges
        for i in range(1, hops - 1):
            lines.append(f"  ?n{i} ?p{i+1} ?n{i+1} .")
            lines.append(f'  FILTER(STRSTARTS(STR(?p{i+1}), "http://www.wikidata.org/prop/direct/"))')
            lines.append(f'  FILTER(STRSTARTS(STR(?n{i+1}), "http://www.wikidata.org/entity/Q"))')

        # Final edge reaches qid2
        lines.append(f"  ?n{hops - 1} ?p{hops} wd:{qid2} .")
        lines.append(f'  FILTER(STRSTARTS(STR(?p{hops}), "http://www.wikidata.org/prop/direct/"))')

        # Mild anti-cycle constraints
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
    Return all DISTINCT first-hop property IDs on qid1 that can reach qid2.

    Example output:
      ["P19", "P27"]
    """
    query = build_first_hop_property_query(qid1, qid2, max_hops)
    data = run_sparql(query)
    bindings = data.get("results", {}).get("bindings", [])

    pids = []
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
    Skip rows that do not look like real QA rows.

    Your CSV contains timing_summary rows with empty question / answer fields.
    """
    if not question and not gold_answer and not formatted:
        return True
    return False


def process_row(row: pd.Series,
                existence_cache: Dict[str, bool],
                label_cache: Dict[str, str]) -> Dict[str, Any]:
    """
    Process one input row and return one output record.

    This is the core per-row logic.
    """
    file_path = normalize_text(row.get("file_path"))
    question = normalize_text(row.get("question"))
    gold_answer = normalize_text(row.get("gold_answer"))
    answer = normalize_text(row.get("answer"))
    formatted = normalize_text(row.get("formatted"))

    # Initialize output record with only the requested kept columns.
    out = {
        "file_path": file_path,
        "question": question,
        "gold_answer": gold_answer,
        "answer": answer,
        "formatted": formatted,
        "inconsistency_taxonomy": "",
        "question_entities": "",
        "gold_answer_entities": "",
        "property_number": "",
        "property_name": "",
    }

    # Skip empty/timing-summary rows quietly.
    if should_skip_row(question, gold_answer, formatted):
        return out

    # ---------------------------------------------------------------------
    # 1) Extract question-side entities.
    # ---------------------------------------------------------------------
    question_qids = extract_question_entities(question, formatted, existence_cache)

    # ---------------------------------------------------------------------
    # 2) Extract gold-answer-side entities.
    # ---------------------------------------------------------------------
    answer_qids = extract_gold_answer_entities(gold_answer, existence_cache)

    # Format them as "label (QID)".
    out["question_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in question_qids
    )
    out["gold_answer_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in answer_qids
    )

    # ---------------------------------------------------------------------
    # 3) Missing-node logic.
    #
    # We treat a row as missing-node if:
    #   - question text exists but no question entity could be linked
    #   - gold_answer is non-numeric and no answer entity could be linked
    #
    # We do NOT force numeric answers into missing-node.
    # ---------------------------------------------------------------------
    answer_is_numeric_like = looks_numeric_like(gold_answer)

    missing_question_entity = (question != "" and len(question_qids) == 0)
    missing_answer_entity = (gold_answer != "" and not answer_is_numeric_like and len(answer_qids) == 0)

    if missing_question_entity or missing_answer_entity:
        out["inconsistency_taxonomy"] = "missing node (entity)"
        return out

    # ---------------------------------------------------------------------
    # 4) If answer is numeric-like and no answer entity exists, we stop here.
    #    We do not assign missing edge because there is no entity pair to test.
    # ---------------------------------------------------------------------
    if answer_is_numeric_like and len(answer_qids) == 0:
        return out

    # ---------------------------------------------------------------------
    # 5) If both sides have entities, test all pair combinations and aggregate
    #    all matched first-hop properties.
    # ---------------------------------------------------------------------
    aggregated_pids: List[str] = []

    for q_qid in question_qids:
        for a_qid in answer_qids:
            try:
                pids = find_connecting_first_hop_pids(q_qid, a_qid, MAX_HOPS)
                aggregated_pids.extend(pids)
            except Exception:
                # If one pair fails due to endpoint/network issues, continue.
                # We do not fail the whole row immediately.
                continue

    aggregated_pids = dedupe_preserve_order(aggregated_pids)

    if len(aggregated_pids) == 0:
        out["inconsistency_taxonomy"] = "missing edge (triplet)"
        return out

    out["property_number"] = len(aggregated_pids)
    out["property_name"] = ";".join(
        format_pid_with_label(pid, label_cache) for pid in aggregated_pids
    )

    return out


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    """
    Main script entry point.

    Steps:
      1) Load CSV
      2) Process each row with tqdm
      3) Build output DataFrame
      4) Save output CSV
    """
    print(f"Reading input CSV: {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH)

    # Live-call caches to reduce repeated traffic.
    existence_cache: Dict[str, bool] = {}
    label_cache: Dict[str, str] = {}

    output_records: List[Dict[str, Any]] = []

    # tqdm progress bar for row processing.
    for _, row in tqdm(df.iterrows(), total=len(df), desc="processing rows"):
        record = process_row(row, existence_cache, label_cache)
        output_records.append(record)

    out_df = pd.DataFrame(output_records)

    # Enforce final output column order exactly.
    final_columns = [
        "file_path",
        "question",
        "gold_answer",
        "answer",
        "formatted",
        "inconsistency_taxonomy",
        "question_entities",
        "gold_answer_entities",
        "property_number",
        "property_name",
    ]
    out_df = out_df[final_columns]

    print(f"Writing output CSV: {OUTPUT_CSV_PATH}")
    out_df.to_csv(OUTPUT_CSV_PATH, index=False)

    print("Done.")
    print(f"Output saved to: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()