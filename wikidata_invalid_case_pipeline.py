#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
# wikidata_invalid_case_pipeline_v5.py
# =============================================================================
#
# PURPOSE
# -------
# This script reads an input CSV and outputs a refined CSV with these columns:
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
# MAIN CHANGES IN THIS VERSION
# ----------------------------
# 1. STRICT QUESTION ENTITY EXTRACTION
#    We now try to extract ALL obvious entity mentions from the question, not just
#    one main topic entity.
#
#    Example:
#      "What is the name of the film in the Bridget Jones series that was not
#       directed by Sharon Maguire"
#
#    should extract:
#      - Bridget Jones
#      - Sharon Maguire
#
#    rather than only "Bridget Jones".
#
# 2. STRICTNESS
#    Question entities must be clearly supported by the actual question text.
#    We do NOT blindly trust all QIDs from `formatted`.
#
# 3. ALL QUESTION ENTITIES MUST SUPPORT THE ANSWER
#    When checking a gold-answer entity against the question entities, we require
#    that the gold answer connect to ALL extracted question entities.
#
#    In practice:
#      - for each question entity q_i
#      - try to find a path q_i -> ... -> gold_answer_entity
#      - only keep an answer entity if all question entities succeed
#
# 4. DATE ANSWER FALLBACK
#    If gold_answer is date-like, we try direct date properties on ALL extracted
#    question entities:
#      - publication date (P577)
#      - start time (P580)
#      - end time (P582)
#      - point in time (P585)
#
#    Missing node (entity) is only assigned after these try-outs fail.
#
#
# DEPENDENCIES
# ------------
#   pip install pandas requests tqdm
#
#
# HOW TO RUN
# ----------
#   python wikidata_invalid_case_pipeline_v5.py
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

INPUT_CSV_PATH = "all_invalid_cases_first20.csv"
OUTPUT_CSV_PATH = "wikidata_pipeline_output_v5.csv"

LABEL_LANGUAGE = "en"
MAX_HOPS = 2

USER_AGENT = "WikidataInvalidCasePipelineV5/1.0 (Python requests; contact: your-email@example.com)"
HEADERS_JSON = {"User-Agent": USER_AGENT, "Accept": "application/json"}

WBSEARCH_API = "https://www.wikidata.org/w/api.php"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"

HTTP_TIMEOUT = 25
RETRIES = 2

DATE_PID_TO_NAME = {
    "P577": "publication date",
    "P580": "start time",
    "P582": "end time",
    "P585": "point in time",
}
DATE_PIDS = list(DATE_PID_TO_NAME.keys())

# A small stopword list used in text normalization / support checks.
STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "from", "to", "by", "with", "and",
    "or", "at", "as", "is", "was", "were", "be", "been", "being", "what", "which",
    "who", "where", "when", "how", "many", "much", "did", "does", "do", "name",
    "year", "first", "last", "game", "play", "played", "team", "series", "song",
    "film", "movie", "tv", "show", "season", "cover", "photo", "location", "not",
    "directed", "written", "performed", "singer", "author", "country", "position",
    "number", "member", "members", "won", "win", "made", "did", "their", "his",
    "her", "its", "that", "this", "these", "those", "there", "whereas", "while"
}

# Words that should not be treated as standalone obvious entity names.
QUESTION_WORDS = {
    "what", "which", "who", "where", "when", "why", "how", "name", "film",
    "movie", "series", "song", "country", "team", "year"
}


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
    """Remove duplicates while preserving order."""
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def normalize_for_match(text: str) -> str:
    """
    Normalize text for containment / overlap matching.

    Lowercase, remove punctuation, collapse whitespace.
    """
    text = text.lower()
    text = re.sub(r"[_/\\|]+", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_match(text: str) -> List[str]:
    """
    Tokenize normalized text and remove simple stopwords.
    """
    text = normalize_for_match(text)
    toks = [t for t in text.split() if t and t not in STOPWORDS]
    return toks


def looks_numeric_like(text: str) -> bool:
    """
    Heuristic detector for numeric/date/quantity-like answers.
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


def looks_date_like(text: str) -> bool:
    """
    More specific detector for date-like answers.
    """
    t = text.strip().lower()
    if not t:
        return False

    if re.fullmatch(r"\d{4}", t):
        return True

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return True

    if re.search(
        r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
        t
    ):
        return True

    return False


def year_from_text(text: str) -> Optional[str]:
    """Extract a 4-digit year if present."""
    m = re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", text)
    return m.group(1) if m else None


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
    """Get label for QID or PID."""
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


def get_aliases_for_qid(qid: str, alias_cache: Dict[str, List[str]], language: str = LABEL_LANGUAGE) -> List[str]:
    """
    Fetch aliases for a QID.

    Used to test whether a candidate entity is really supported by question text.
    """
    if qid in alias_cache:
        return alias_cache[qid]

    params = {
        "action": "wbgetentities",
        "ids": qid,
        "props": "labels|aliases",
        "languages": language,
        "format": "json",
    }

    aliases: List[str] = []
    try:
        resp = safe_get(WBSEARCH_API, params=params, headers={"User-Agent": USER_AGENT})
        data = resp.json()
        ent = data.get("entities", {}).get(qid, {})
        if "labels" in ent and language in ent["labels"]:
            aliases.append(ent["labels"][language]["value"])
        for a in ent.get("aliases", {}).get(language, []):
            aliases.append(a["value"])
    except Exception:
        aliases = [qid]

    aliases = dedupe_preserve_order([a for a in aliases if a])
    alias_cache[qid] = aliases
    return aliases


def format_qid_with_label(qid: str, label_cache: Dict[str, str]) -> str:
    """Format QID as name (QID)."""
    return f"{get_label_for_id(qid, label_cache)} ({qid})"


def format_pid_with_label(pid: str, label_cache: Dict[str, str]) -> str:
    """Format PID as label (PID)."""
    return f"{get_label_for_id(pid, label_cache)} ({pid})"


# =============================================================================
# EXPLICIT ID EXTRACTION
# =============================================================================

def extract_explicit_qids(text: str) -> List[str]:
    """Extract explicit QIDs from text."""
    if not text:
        return []
    matches = re.findall(r"\bQ[1-9]\d*\b", text, flags=re.IGNORECASE)
    return dedupe_preserve_order([m.upper() for m in matches])


def extract_qids_from_formatted(formatted: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract QIDs from formatted and keep only existing ones.
    """
    qids = extract_explicit_qids(formatted)
    qids = [qid for qid in qids if qid_exists(qid, existence_cache)]
    return dedupe_preserve_order(qids)


# =============================================================================
# STRICT QUESTION ENTITY SUPPORT
# =============================================================================

def question_supports_entity(question: str, qid: str, label_cache: Dict[str, str], alias_cache: Dict[str, List[str]]) -> bool:
    """
    Strictly check whether an entity is supported by question text.

    Accept if:
      - full normalized label/alias appears in normalized question text
      - OR all informative tokens of a label/alias appear in question text
    """
    q_norm = normalize_for_match(question)
    q_tokens = set(tokenize_for_match(question))
    if not q_norm:
        return False

    names = get_aliases_for_qid(qid, alias_cache)
    if not names:
        names = [get_label_for_id(qid, label_cache)]

    for name in names:
        n_norm = normalize_for_match(name)
        n_tokens = tokenize_for_match(name)

        if not n_norm:
            continue

        if n_norm in q_norm:
            return True

        if n_tokens and all(tok in q_tokens for tok in n_tokens):
            return True

    return False


# =============================================================================
# STRICT OBVIOUS-NAME EXTRACTION FROM QUESTION
# =============================================================================

def extract_capitalized_name_spans(question: str) -> List[str]:
    """
    Extract obvious name-like spans from the question.

    This targets explicit names such as:
      - Sharon Maguire
      - Tim Howard
      - Ellen White
      - Alex Morgan
      - Megan Rapinoe
      - Bridget Jones

    It is intentionally strict and biased toward obvious proper names.

    Approach:
      - find contiguous sequences of capitalized tokens / initials / apostrophe names
      - discard leading WH words and generic words
      - keep spans of 1+ capitalized tokens, but prefer multi-token spans
    """
    if not question:
        return []

    # Preserve original capitalization; lightly clean punctuation.
    text = re.sub(r"[“”\"`]", "", question)
    text = re.sub(r"[\(\)\[\]\{\}:;!?]", " ", text)
    tokens = text.split()

    spans: List[str] = []
    current: List[str] = []

    def is_name_token(tok: str) -> bool:
        # Accept tokens like Sharon, Maguire, O'Neil, U.S., McDonald
        return bool(re.fullmatch(r"[A-Z][A-Za-z'’-]*\.?", tok))

    def flush_current():
        nonlocal current, spans
        if current:
            span = " ".join(current).strip(" ,.")
            if span:
                spans.append(span)
        current = []

    for tok in tokens:
        clean_tok = tok.strip(" ,.")
        if is_name_token(clean_tok):
            current.append(clean_tok)
        else:
            flush_current()

    flush_current()

    # Filter bad spans
    filtered: List[str] = []
    for span in spans:
        norm = normalize_for_match(span)
        toks = norm.split()
        if not toks:
            continue

        # Drop spans that start with question words or are entirely generic.
        if toks[0] in QUESTION_WORDS:
            continue

        # Drop one-token generic nouns.
        if len(toks) == 1 and toks[0] in {"film", "series", "song", "team", "country", "year"}:
            continue

        filtered.append(span)

    # Prefer longer spans first; still preserve deduped order later.
    filtered = sorted(filtered, key=lambda x: (-len(x.split()), question.find(x)))
    return dedupe_preserve_order(filtered)


def extract_candidate_question_surfaces(question: str) -> List[str]:
    """
    Build candidate question surfaces for live search.

    Priority:
      1) obvious capitalized name spans
      2) full question
      3) stripped question
      4) split fragments

    The goal is to capture ALL obvious names in the question.
    """
    q = question.strip()
    if not q:
        return []

    q = re.sub(r"\s+", " ", q)
    candidates: List[str] = []

    # 1) obvious name-like spans first
    candidates.extend(extract_capitalized_name_spans(q))

    # 2) full question sometimes helps as a fallback
    candidates.append(q.strip(" ?"))

    # 3) remove leading WH template
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
        " where ", " that ", " which ", " who ", " whose ", " not directed by "
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

    Handles multiple entity answers separated by commas / 'and' / '&'.
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

    Policy:
      - explicit QID wins
      - otherwise first existing search hit wins
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

def extract_question_entities(
    question: str,
    formatted: str,
    existence_cache: Dict[str, bool],
    label_cache: Dict[str, str],
    alias_cache: Dict[str, List[str]],
) -> List[str]:
    """
    Extract ALL strict question-side entities.

    Priority:
      1) formatted QIDs that are text-supported by the question
      2) explicit QIDs in the question
      3) live-search on obvious question name spans and other candidate surfaces

    This version is stricter and multi-entity aware.
    """
    out_qids: List[str] = []

    # 1) formatted QIDs, but ONLY if supported by the question text
    formatted_qids = extract_qids_from_formatted(formatted, existence_cache)
    for qid in formatted_qids:
        if question_supports_entity(question, qid, label_cache, alias_cache):
            out_qids.append(qid)

    # 2) explicit QIDs directly in the question
    for qid in extract_explicit_qids(question):
        if qid_exists(qid, existence_cache):
            out_qids.append(qid)

    # 3) live-search candidate question surfaces, keeping only strict matches
    for cand in extract_candidate_question_surfaces(question):
        qid = pick_best_qid_for_surface(cand, existence_cache)
        if qid and question_supports_entity(question, qid, label_cache, alias_cache):
            out_qids.append(qid)

    return dedupe_preserve_order(out_qids)


def extract_gold_answer_entities(gold_answer: str, existence_cache: Dict[str, bool]) -> List[str]:
    """
    Extract gold-answer-side entities.

    Date-like and numeric-like answers are not forced into entity linking.
    """
    gold_answer = gold_answer.strip()
    if not gold_answer:
        return []

    if looks_date_like(gold_answer) or looks_numeric_like(gold_answer):
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
# SPARQL HELPERS
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


# =============================================================================
# ENTITY-TO-ENTITY PATH SEARCH
# =============================================================================

def build_exact_path_query(qid1: str, qid2: str, hops: int) -> str:
    """
    Build a SPARQL query for one outward path of exactly `hops` edges from qid1 to qid2.
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
    Convert one SPARQL result row to path edges:
      [(source_qid, pid, target_qid), ...]
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
    Convert a path into a readable string:
      Entity (QID) -> property (PID) -> Entity (QID) -> ...
    """
    if not path_edges:
        return ""

    parts = [format_qid_with_label(path_edges[0][0], label_cache)]
    for _, pid, target_qid in path_edges:
        parts.append(format_pid_with_label(pid, label_cache))
        parts.append(format_qid_with_label(target_qid, label_cache))

    return " -> ".join(parts)


def get_first_hop_pids_from_path(path_edges: List[Tuple[str, str, str]]) -> List[str]:
    """
    Extract first-hop PID from a found entity path.
    """
    if not path_edges:
        return []
    return [path_edges[0][1]]


# =============================================================================
# DATE PROPERTY SEARCH
# =============================================================================

def build_date_property_query(qid: str, pid: str) -> str:
    """
    Query a direct date property for one entity.
    """
    return f"""
    SELECT ?value WHERE {{
      wd:{qid} wdt:{pid} ?value .
    }}
    """


def value_matches_gold_date(value_str: str, gold_answer: str) -> bool:
    """
    Decide whether a Wikidata date/time value matches the gold answer.

    Policy:
      - if gold answer has a year, compare by year
      - otherwise do substring-based normalized matching
    """
    gold = gold_answer.strip().lower()
    value = value_str.strip().lower()

    gold_year = year_from_text(gold)
    value_year = year_from_text(value)

    if gold_year:
        return value_year == gold_year

    return gold in value or value in gold


def find_date_property_matches(
    question_qids: List[str],
    gold_answer: str,
    label_cache: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """
    Try matching date-like gold answers using date properties on ALL question entities.

    Returns:
      matched_pids
      readable_paths
    """
    matched_pids: List[str] = []
    readable_paths: List[str] = []

    for qid in question_qids:
        q_label = format_qid_with_label(qid, label_cache)

        for pid in DATE_PIDS:
            try:
                data = run_sparql(build_date_property_query(qid, pid))
            except Exception:
                continue

            bindings = data.get("results", {}).get("bindings", [])
            for row in bindings:
                value = row.get("value", {}).get("value", "")
                if value and value_matches_gold_date(value, gold_answer):
                    matched_pids.append(pid)
                    readable_paths.append(
                        f"{q_label} -> {format_pid_with_label(pid, label_cache)} -> {value}"
                    )

    return dedupe_preserve_order(matched_pids), dedupe_preserve_order(readable_paths)


# =============================================================================
# ROW PROCESSING
# =============================================================================

def should_skip_row(question: str, gold_answer: str, formatted: str) -> bool:
    """Skip rows that are effectively empty / non-QA."""
    return (not question and not gold_answer and not formatted)


def all_question_entities_connect_to_answer(
    question_qids: List[str],
    answer_qid: str,
    label_cache: Dict[str, str],
) -> Tuple[bool, List[str], List[str]]:
    """
    Require that ALL extracted question entities connect to the answer entity.

    Returns:
      success
      first_hop_pids
      readable_paths

    success is True only if every question entity has at least one path to answer_qid.
    """
    all_first_hop_pids: List[str] = []
    all_readable_paths: List[str] = []

    if not question_qids:
        return False, [], []

    for q_qid in question_qids:
        try:
            path_edges = find_one_outward_path(q_qid, answer_qid, MAX_HOPS)
        except Exception:
            path_edges = None

        if not path_edges:
            return False, [], []

        all_first_hop_pids.extend(get_first_hop_pids_from_path(path_edges))
        all_readable_paths.append(path_to_readable_string(path_edges, label_cache))

    return True, dedupe_preserve_order(all_first_hop_pids), dedupe_preserve_order(all_readable_paths)


def process_row(
    row: pd.Series,
    existence_cache: Dict[str, bool],
    label_cache: Dict[str, str],
    alias_cache: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Process one row into the final output format.

    Rules:
      - extract ALL strict question entities
      - if gold_answer is date-like, try date properties on question entities
      - otherwise, require the answer entity to connect to ALL question entities
      - assign missing node only after these try-outs
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

    # -----------------------------------------------------------------
    # Step 1: Extract ALL strict question entities
    # -----------------------------------------------------------------
    question_qids = extract_question_entities(
        question=question,
        formatted=formatted,
        existence_cache=existence_cache,
        label_cache=label_cache,
        alias_cache=alias_cache,
    )

    # -----------------------------------------------------------------
    # Step 2: Extract gold-answer entities
    # -----------------------------------------------------------------
    answer_qids = extract_gold_answer_entities(gold_answer, existence_cache)

    out["question_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in question_qids
    )
    out["gold_answer_entities"] = ";".join(
        format_qid_with_label(qid, label_cache) for qid in answer_qids
    )

    # -----------------------------------------------------------------
    # Step 3: Date-answer fallback
    # -----------------------------------------------------------------
    answer_is_date_like = looks_date_like(gold_answer)
    if answer_is_date_like and question_qids:
        matched_pids, readable_paths = find_date_property_matches(
            question_qids=question_qids,
            gold_answer=gold_answer,
            label_cache=label_cache,
        )
        if matched_pids:
            out["property_number"] = len(matched_pids)
            out["property_name"] = ";".join(
                format_pid_with_label(pid, label_cache) for pid in matched_pids
            )
            out["connection_path"] = " || ".join(readable_paths)
            return out

    # -----------------------------------------------------------------
    # Step 4: Normal entity-to-entity checking
    #         Require ALL question entities to connect to the answer entity
    # -----------------------------------------------------------------
    if question_qids and answer_qids:
        successful_answer_qids: List[str] = []
        aggregated_pids: List[str] = []
        aggregated_paths: List[str] = []

        for a_qid in answer_qids:
            success, pids, paths = all_question_entities_connect_to_answer(
                question_qids=question_qids,
                answer_qid=a_qid,
                label_cache=label_cache,
            )
            if success:
                successful_answer_qids.append(a_qid)
                aggregated_pids.extend(pids)
                aggregated_paths.extend(paths)

        successful_answer_qids = dedupe_preserve_order(successful_answer_qids)
        aggregated_pids = dedupe_preserve_order(aggregated_pids)
        aggregated_paths = dedupe_preserve_order(aggregated_paths)

        if successful_answer_qids:
            # Restrict displayed gold answer entities to only the ones that passed
            out["gold_answer_entities"] = ";".join(
                format_qid_with_label(qid, label_cache) for qid in successful_answer_qids
            )
            out["property_number"] = len(aggregated_pids)
            out["property_name"] = ";".join(
                format_pid_with_label(pid, label_cache) for pid in aggregated_pids
            )
            out["connection_path"] = " || ".join(aggregated_paths)
            return out

        # Both question entities and answer entities exist, but the answer
        # could not be connected to ALL question entities.
        out["inconsistency_taxonomy"] = "missing edge (triplet)"
        return out

    # -----------------------------------------------------------------
    # Step 5: Assign missing node only AFTER all try-outs
    # -----------------------------------------------------------------
    answer_is_numeric_like = looks_numeric_like(gold_answer)

    missing_question_entity = (question != "" and len(question_qids) == 0)
    missing_answer_entity = (
        gold_answer != ""
        and not answer_is_numeric_like
        and not answer_is_date_like
        and len(answer_qids) == 0
    )

    if missing_question_entity or missing_answer_entity:
        out["inconsistency_taxonomy"] = "missing node (entity)"

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
    alias_cache: Dict[str, List[str]] = {}

    output_records: List[Dict[str, Any]] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="processing rows"):
        output_records.append(process_row(row, existence_cache, label_cache, alias_cache))

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