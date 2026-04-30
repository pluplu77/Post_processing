#!/usr/bin/env python3
"""
derive_connection_paths_from_sparql_results_v4_literals.py

Purpose
-------
Read an input CSV containing SPARQL queries and their result tables, then derive
connection paths directly from the two columns:

    result
    sparql

The output CSV keeps the input question and gold answer as the first two columns,
then writes this schema:

    question,
    gold_answer,
    Entity_1,
    Entity_2,
    Connection_Path,
    Property_Number,
    Property_list,
    Qualifier_Number,
    Qualifier_list

Main idea
---------
The script does not query Wikidata to discover paths. It derives paths from the
SPARQL text itself.

Example entity-valued query:

    SELECT ?entity WHERE {
      wd:Q1646482 wdt:P800 ?entity .
    }

If the result table binds ?entity to Hamilton (wd:Q84323848), the derived path is:

    Lin-Manuel Miranda (Q1646482)->notable work (P800)->Hamilton (Q84323848)

Example literal-valued query:

    SELECT ?duration WHERE {
      wd:Q18758167 wdt:P2047 ?duration
    }

If the result table binds ?duration to 252.0 (xsd:decimal), the output is:

    Entity_1:        Love Me like You Do (Q18758167)
    Entity_2:        xsd:decimal
    Connection_Path: Love Me like You Do (Q18758167)->duration (P2047)->xsd:decimal

Label strategy
--------------
Labels are resolved in this order:

1. Labels already present in the `result` table.
   For example:
       Hamilton (wd:Q84323848)
       Hamilton (lang:en)

2. Labels already seen in other rows of the same input CSV.

3. Wikidata wbgetentities API, only for IDs still missing labels.
   This includes:
       QIDs
       PIDs
       qualifier PIDs

4. Raw ID fallback.
   If the API is unavailable, the script still writes rows using raw IDs.

Literal strategy
----------------
If a selected result variable is not a Wikidata entity but has a datatype such as:

    252.0 (xsd:decimal)
    2015-01-01 (xsd:dateTime)

then the script uses the datatype as the second endpoint. This allows literal
paths to be written instead of skipped.

Supported SPARQL patterns
-------------------------
This is a practical parser for common Wikidata query patterns, not a full SPARQL
engine. It supports:

1. Direct truthy properties:
       wd:Q1 wdt:P123 ?x
       ?x wdt:P456 wd:Q2
       ?x wdt:P789 ?y

2. Statement/qualifier patterns:
       wd:Q1 p:P1346 ?statement .
       ?statement ps:P1346 ?winner .
       ?statement pq:P1686 ?work .

   This becomes:
       Q1 -> P1346 -> winner -> P1686 -> work

   P1346 is counted in Property_list.
   P1686 is counted in Qualifier_list.

3. Multiple paths between the same pair. These are formatted as:
       Path1: ...
       Path2: ...

Known limitations
-----------------
The script does not fully implement SPARQL. It intentionally focuses on common
Wikidata QA patterns. It may skip queries that rely on complex UNION semantics,
property paths such as wdt:P31/wdt:P279*, BIND-created IRIs, VALUES-only data,
subqueries, aggregations, or SERVICE-only label lookups.

How to run
----------
1. Install dependencies:
       pip install requests tqdm

   `tqdm` is optional. If it is not installed, the script still runs.

2. Put this script next to your input CSV or edit INPUT_CSV_PATH below.

3. Run:
       python derive_connection_paths_from_sparql_results_v4_literals.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

try:
    import requests
except Exception:  # pragma: no cover - handled at runtime
    requests = None  # type: ignore

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV_PATH = "complexqa_all_valid_cases_with_taxonomy.csv"
OUTPUT_CSV_PATH = "complexqa_derived__all_valid_cases_with_taxonomy.csv"
SKIPPED_REPORT_CSV_PATH = "complexqa_connection_skipped_report.csv"

LABEL_LANGUAGE = "en"

# If True, missing QID/PID labels are fetched from Wikidata. Labels found in the
# input result table are always preferred and are not fetched again.
USE_WIKIDATA_API_FOR_MISSING_LABELS = True

# Batch size for wbgetentities. 40 is conservative and API-friendly.
WIKIDATA_LABEL_BATCH_SIZE = 40

# Polite pause between label API batches.
WIKIDATA_API_SLEEP_SECONDS = 0.05

WBGETENTITIES_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "SPARQLResultConnectionPathDeriver/4.0 "
        "(Python requests; contact: your-email@example.com)"
    )
}

# Maximum number of edges to traverse in the instantiated graph from one result
# row. Most QA paths are short.
MAX_PATH_EDGES = 5

# If True, write a blank output row for source rows where no path was derived.
# Usually False is cleaner because the skipped report explains what happened.
WRITE_UNRESOLVED_ROWS = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Binding:
    """
    Concrete value for a SPARQL result variable.

    kind:
        "qid"     -> Wikidata item, e.g. Q84323848
        "pid"     -> Wikidata property, e.g. P800
        "literal" -> non-entity value, represented by a datatype such as xsd:decimal

    value:
        QID, PID, or literal endpoint token. For literals this is usually the
        datatype, e.g. xsd:decimal.

    label:
        Human-readable label from the result cell when available. For literals,
        this is usually the literal lexical value, e.g. 252.0.
    """

    kind: str
    value: str
    label: str = ""
    datatype: str = ""


@dataclass(frozen=True)
class Edge:
    """
    One directed edge parsed from the SPARQL.

    Before instantiation, source and target may be either:
        - QIDs such as Q1646482
        - variable names without '?', such as entity

    After instantiation, source is normally a QID and target may be a QID or a
    literal endpoint such as xsd:decimal.

    edge_type:
        "property"  -> normal direct/main property
        "qualifier" -> qualifier property from pq:Pxxx
    """

    source: str
    pid: str
    target: str
    edge_type: str


@dataclass(frozen=True)
class PathResult:
    """One concrete output path."""

    question: str
    gold_answer: str
    entity_1: str
    entity_2: str
    path_tokens: Tuple[str, ...]
    property_ids: Tuple[str, ...]
    qualifier_ids: Tuple[str, ...]


# ============================================================
# BASIC HELPERS
# ============================================================

def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def is_qid(value: str) -> bool:
    return bool(re.fullmatch(r"Q[1-9]\d*", str(value or "")))


def is_pid(value: str) -> bool:
    return bool(re.fullmatch(r"P[1-9]\d*", str(value or "")))


def is_wikidata_id(value: str) -> bool:
    return is_qid(value) or is_pid(value)


def is_literal_endpoint(value: str) -> bool:
    """
    Return True for literal endpoints.

    Supports both compact datatype endpoints:
        xsd:decimal

    and full literal endpoints:
        252.0 (xsd:decimal)
        2015-01-01 (xsd:dateTime)
    """
    value = str(value or "").strip()

    if value == "literal":
        return True

    # Compact datatype only, e.g. xsd:decimal
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]*:[A-Za-z_][A-Za-z0-9_.-]*", value):
        return True

    # Full literal with datatype, e.g. 252.0 (xsd:decimal)
    if re.fullmatch(
        r".+\s+\([A-Za-z_][A-Za-z0-9_.-]*:[A-Za-z_][A-Za-z0-9_.-]*\)",
        value,
    ):
        return True

    return False


def normalize_sparql_token(token: str) -> str:
    """
    Normalize simple SPARQL tokens.

    Examples:
        wd:Q1       -> Q1
        wdt:P31     -> P31
        p:P1346     -> P1346
        ps:P1346    -> P1346
        pq:P1686    -> P1686
        ?entity     -> entity
        <.../Q1>    -> Q1, if the IRI ends in Q1
    """
    token = str(token or "").strip().rstrip(".;,")

    iri_match = re.search(r"/(Q[1-9]\d*|P[1-9]\d*)>?$", token)
    if iri_match:
        return iri_match.group(1)

    for prefix in ("wd:", "wdt:", "p:", "ps:", "pq:", "pr:", "prov:"):
        if token.startswith(prefix):
            return token[len(prefix):]

    if token.startswith("?"):
        return token[1:]

    return token


def strip_sparql_comments(sparql: str) -> str:
    """Remove line comments. This simple heuristic does not parse string literals."""
    lines: List[str] = []
    for line in str(sparql or "").splitlines():
        lines.append(re.sub(r"#.*$", "", line))
    return "\n".join(lines)


# ============================================================
# LABEL STORE
# ============================================================

class LabelStore:
    """
    Store labels found in the CSV and fetch missing labels from Wikidata.

    Priority order:
        1. labels found in the current result row
        2. labels found in previous result rows
        3. Wikidata API fallback
        4. raw ID fallback
    """

    def __init__(self, language: str = LABEL_LANGUAGE) -> None:
        self.language = language
        self.global_labels: Dict[str, str] = {}
        self.missing_cache: Set[str] = set()

    def add_label(self, entity_id: str, label: str) -> None:
        entity_id = str(entity_id or "").strip().upper()
        label = str(label or "").strip()
        if not entity_id or not label:
            return
        if not is_wikidata_id(entity_id):
            return
        if label == entity_id:
            return
        self.global_labels.setdefault(entity_id, label)

    def add_labels(self, labels: Dict[str, str]) -> None:
        for entity_id, label in labels.items():
            self.add_label(entity_id, label)

    def fetch_missing_labels(self, ids: Iterable[str]) -> None:
        """Fetch missing QID/PID labels in batches."""
        if not USE_WIKIDATA_API_FOR_MISSING_LABELS:
            return
        if requests is None:
            print("Warning: requests is not installed; using raw IDs for missing labels.", file=sys.stderr)
            return

        ids_to_fetch = [
            item_id
            for item_id in unique_preserve_order(ids)
            if is_wikidata_id(item_id)
            and item_id not in self.global_labels
            and item_id not in self.missing_cache
        ]

        for start in range(0, len(ids_to_fetch), WIKIDATA_LABEL_BATCH_SIZE):
            batch = ids_to_fetch[start:start + WIKIDATA_LABEL_BATCH_SIZE]
            if not batch:
                continue

            params = {
                "action": "wbgetentities",
                "ids": "|".join(batch),
                "props": "labels",
                "languages": self.language,
                "format": "json",
            }

            try:
                response = requests.get(
                    WBGETENTITIES_API,
                    params=params,
                    headers=HEADERS,
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()

                entities = data.get("entities", {})
                for item_id in batch:
                    entity_data = entities.get(item_id, {})
                    labels = entity_data.get("labels", {})
                    value = labels.get(self.language, {}).get("value", "")
                    if value:
                        self.global_labels[item_id] = value
                    else:
                        self.missing_cache.add(item_id)

            except Exception as exc:
                print(f"Warning: failed to fetch labels for batch {batch}: {exc}", file=sys.stderr)
                self.missing_cache.update(batch)

            time.sleep(WIKIDATA_API_SLEEP_SECONDS)

    def label_for(self, entity_id: str, row_labels: Optional[Dict[str, str]] = None) -> str:
        entity_id = str(entity_id or "").strip().upper()
        if row_labels and entity_id in row_labels:
            return row_labels[entity_id]
        if entity_id in self.global_labels:
            return self.global_labels[entity_id]
        return entity_id

    def readable_id(self, token: str, row_labels: Optional[Dict[str, str]] = None) -> str:
        """
        Format QIDs/PIDs as Label (ID). Literal endpoints such as xsd:decimal are
        returned as-is.
        """
        token = str(token or "").strip()
        if is_wikidata_id(token):
            token_upper = token.upper()
            label = self.label_for(token_upper, row_labels)
            return f"{label} ({token_upper})"
        return token


# ============================================================
# RESULT TABLE PARSING
# ============================================================

def split_markdown_row(line: str) -> List[str]:
    """Split a simple Markdown table row."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def parse_result_cell(cell: str) -> Dict[str, str]:
    """
    Parse one result table cell.

    Supported examples:
        Hamilton (wd:Q84323848)
        Hamilton (lang:en)
        winner (wdt:P1346)
        252.0 (xsd:decimal)
        2015-01-01 (xsd:dateTime)
        Q123
        P456
    """
    cell = str(cell or "").strip()

    parsed = {
        "raw": cell,
        "qid": "",
        "pid": "",
        "id": "",
        "label": "",
        "literal_value": "",
        "datatype": "",
    }

    # Entity/property with Wikidata-style prefix inside parentheses.
    id_match = re.search(r"\((?:wd|wdt|p|ps|pq):(Q[1-9]\d*|P[1-9]\d*)\)", cell)
    if id_match:
        item_id = id_match.group(1).upper()
        label = re.sub(
            r"\s*\((?:wd|wdt|p|ps|pq):(?:Q[1-9]\d*|P[1-9]\d*)\)\s*",
            "",
            cell,
        ).strip()
        parsed["id"] = item_id
        parsed["label"] = label
        if is_qid(item_id):
            parsed["qid"] = item_id
        elif is_pid(item_id):
            parsed["pid"] = item_id
        return parsed

    # Literal with datatype, e.g. 252.0 (xsd:decimal).
    datatype_match = re.match(
        r"^(?P<value>.*?)\s*\((?P<datatype>[A-Za-z_][A-Za-z0-9_.-]*:[A-Za-z_][A-Za-z0-9_.-]*)\)\s*$",
        cell,
    )
    if datatype_match:
        parsed["literal_value"] = datatype_match.group("value").strip()
        parsed["datatype"] = datatype_match.group("datatype").strip()
        parsed["label"] = parsed["literal_value"]
        return parsed

    # Language-tagged label cell such as Hamilton (lang:en).
    label_match = re.match(r"(.+?)\s*\(lang:[^)]+\)\s*$", cell)
    if label_match:
        parsed["label"] = label_match.group(1).strip()
        return parsed

    # Bare QID/PID fallback.
    bare_match = re.fullmatch(r"(Q[1-9]\d*|P[1-9]\d*)", cell.upper())
    if bare_match:
        item_id = bare_match.group(1).upper()
        parsed["id"] = item_id
        parsed["label"] = item_id
        if is_qid(item_id):
            parsed["qid"] = item_id
        elif is_pid(item_id):
            parsed["pid"] = item_id
        return parsed

    # Generic literal fallback. We keep this as a literal endpoint named "literal".
    if cell:
        parsed["literal_value"] = cell
        parsed["datatype"] = "literal"
        parsed["label"] = cell

    return parsed


def parse_markdown_result_table(result_text: str) -> List[Dict[str, Dict[str, str]]]:
    """Parse a Markdown table from the `result` column."""
    text = str(result_text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return []

    headers = split_markdown_row(table_lines[0])
    rows: List[Dict[str, Dict[str, str]]] = []

    for line in table_lines[1:]:
        if re.fullmatch(r"\|?\s*[-:\s|]+\s*\|?", line):
            continue
        cells = split_markdown_row(line)
        if len(cells) != len(headers):
            continue

        parsed_row: Dict[str, Dict[str, str]] = {}
        for header, cell in zip(headers, cells):
            parsed_row[header.strip()] = parse_result_cell(cell)
        rows.append(parsed_row)

    return rows


def build_variable_bindings_and_row_labels(
    parsed_result_row: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, Binding], Dict[str, str]]:
    """
    Build variable bindings and row-level labels from one result table row.

    Returns:
        var_bindings:
            variable name -> Binding

        row_labels:
            QID/PID -> label found in this result row

    Important fix:
        A variable whose name ends with Label is NOT automatically ignored. If a
        cell contains a QID, it is treated as an entity even if the variable name
        is ?recordLabel or similar.
    """
    var_bindings: Dict[str, Binding] = {}
    row_labels: Dict[str, str] = {}

    # First pass: bind QIDs/PIDs/literals.
    for column, cell in parsed_result_row.items():
        item_id = cell.get("id") or cell.get("qid") or cell.get("pid") or ""
        label = cell.get("label", "")
        datatype = cell.get("datatype", "")

        if item_id:
            kind = "qid" if is_qid(item_id) else "pid"
            var_bindings[column] = Binding(kind=kind, value=item_id, label=label)
            if label and label != item_id:
                row_labels[item_id] = label
            continue

    
        if datatype:
            # For output endpoints, keep the complete literal value.
            # Example:
            #   252.0 (xsd:decimal) -> 252.0 (xsd:decimal)
            literal_value = cell.get("literal_value", "") or label

            if datatype == "literal":
                literal_endpoint = literal_value or "literal"
            else:
                literal_endpoint = f"{literal_value} ({datatype})" if literal_value else datatype

            var_bindings[column] = Binding(
                kind="literal",
                value=literal_endpoint,
                label=literal_value,
                datatype=datatype,
            )

        

    # Second pass: attach label columns to their base entity variables.
    for column, cell in parsed_result_row.items():
        label = cell.get("label", "")
        if not label:
            continue

        possible_base_columns: List[str] = []
        if column.endswith("Label"):
            possible_base_columns.append(column[:-5])

        if column.lower() == "label":
            qid_columns = [name for name, binding in var_bindings.items() if binding.kind == "qid"]
            if len(qid_columns) == 1:
                possible_base_columns.append(qid_columns[0])

        for base_column in possible_base_columns:
            bound = var_bindings.get(base_column)
            if bound and is_wikidata_id(bound.value):
                row_labels[bound.value] = label

    return var_bindings, row_labels


# ============================================================
# SPARQL PARSING
# ============================================================

def extract_select_variables(sparql: str) -> List[str]:
    """Extract variable names from SELECT ... WHERE."""
    query = strip_sparql_comments(sparql)
    match = re.search(
        r"SELECT\s+(?:DISTINCT\s+)?(?P<select>.*?)\s+WHERE\s*\{",
        query,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return []

    select_part = match.group("select")
    return unique_preserve_order(
        var[1:] for var in re.findall(r"\?[A-Za-z_][A-Za-z0-9_]*", select_part)
    )


def extract_fixed_qids(sparql: str) -> List[str]:
    """Extract fixed wd:Q... anchors from the SPARQL text."""
    return unique_preserve_order(re.findall(r"wd:(Q[1-9]\d*)", str(sparql or "")))


def parse_direct_wdt_edges(sparql: str) -> List[Edge]:
    """
    Parse simple direct truthy triples.

    The object may be a QID or a variable. If the variable is bound to a literal
    datatype in the result table, this later becomes a literal-valued path.
    """
    query = strip_sparql_comments(sparql)
    token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    pattern = re.compile(
        rf"(?P<s>{token})\s+wdt:(?P<p>P[1-9]\d*)\s+(?P<o>{token})(?=\s*(?:[.;}}]|$))",
        flags=re.IGNORECASE,
    )

    edges: List[Edge] = []
    for match in pattern.finditer(query):
        edges.append(
            Edge(
                source=normalize_sparql_token(match.group("s")),
                pid=match.group("p").upper(),
                target=normalize_sparql_token(match.group("o")),
                edge_type="property",
            )
        )
    return edges


def parse_qualifier_edges(sparql: str) -> List[Edge]:
    """
    Parse common Wikidata statement/qualifier patterns.

    Example:
        wd:Q503034 p:P1346 ?statement .
        ?statement ps:P1346 ?winner .
        ?statement pq:P1686 ?work .

    Parsed as:
        Q503034 -> P1346 -> winner
        winner  -> P1686 -> work
    """
    query = strip_sparql_comments(sparql)

    subject_token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    object_token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    statement_token = r"\?[A-Za-z_][A-Za-z0-9_]*"

    claim_pattern = re.compile(
        rf"(?P<subject>{subject_token})\s+p:(?P<mainprop>P[1-9]\d*)\s+(?P<statement>{statement_token})(?=\s*(?:[.;}}]|$))",
        flags=re.IGNORECASE,
    )
    ps_pattern = re.compile(
        rf"(?P<statement>{statement_token})\s+ps:(?P<mainprop>P[1-9]\d*)\s+(?P<mainvalue>{object_token})(?=\s*(?:[.;}}]|$))",
        flags=re.IGNORECASE,
    )
    pq_pattern = re.compile(
        rf"(?P<statement>{statement_token})\s+pq:(?P<qualprop>P[1-9]\d*)\s+(?P<qualvalue>{object_token})(?=\s*(?:[.;}}]|$))",
        flags=re.IGNORECASE,
    )

    claim_by_statement: Dict[str, Tuple[str, str]] = {}
    main_value_by_statement: Dict[str, Tuple[str, str]] = {}
    qualifiers_by_statement: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for match in claim_pattern.finditer(query):
        statement = normalize_sparql_token(match.group("statement"))
        subject = normalize_sparql_token(match.group("subject"))
        mainprop = match.group("mainprop").upper()
        claim_by_statement[statement] = (subject, mainprop)

    for match in ps_pattern.finditer(query):
        statement = normalize_sparql_token(match.group("statement"))
        mainprop = match.group("mainprop").upper()
        mainvalue = normalize_sparql_token(match.group("mainvalue"))
        main_value_by_statement[statement] = (mainprop, mainvalue)

    for match in pq_pattern.finditer(query):
        statement = normalize_sparql_token(match.group("statement"))
        qualprop = match.group("qualprop").upper()
        qualvalue = normalize_sparql_token(match.group("qualvalue"))
        qualifiers_by_statement[statement].append((qualprop, qualvalue))

    edges: List[Edge] = []
    for statement, (subject, claim_mainprop) in claim_by_statement.items():
        if statement not in main_value_by_statement:
            continue
        ps_mainprop, mainvalue = main_value_by_statement[statement]
        if ps_mainprop != claim_mainprop:
            continue

        edges.append(Edge(source=subject, pid=claim_mainprop, target=mainvalue, edge_type="property"))
        for qualprop, qualvalue in qualifiers_by_statement.get(statement, []):
            edges.append(Edge(source=mainvalue, pid=qualprop, target=qualvalue, edge_type="qualifier"))

    return edges


def parse_sparql_edges(sparql: str) -> List[Edge]:
    """Parse all supported edge types from SPARQL and de-duplicate them."""
    edges: List[Edge] = []
    edges.extend(parse_direct_wdt_edges(sparql))
    edges.extend(parse_qualifier_edges(sparql))

    seen: Set[Edge] = set()
    output: List[Edge] = []
    for edge in edges:
        if edge not in seen:
            seen.add(edge)
            output.append(edge)
    return output


# ============================================================
# EDGE INSTANTIATION AND PATH DISCOVERY
# ============================================================

def instantiate_token(token: str, var_bindings: Dict[str, Binding]) -> Optional[str]:
    """Convert a symbolic token into a concrete QID/PID/literal endpoint."""
    token = str(token or "").strip()
    if is_qid(token) or is_pid(token) or is_literal_endpoint(token):
        return token
    binding = var_bindings.get(token)
    if binding:
        return binding.value
    return None


def instantiate_edges(edges: List[Edge], var_bindings: Dict[str, Binding]) -> List[Edge]:
    """Fill variables in parsed edges using the current result row."""
    instantiated: List[Edge] = []

    for edge in edges:
        source = instantiate_token(edge.source, var_bindings)
        target = instantiate_token(edge.target, var_bindings)

        if not source or not target:
            continue
        if not is_pid(edge.pid):
            continue

        # Source nodes in Wikidata triples should be QIDs for this output. Target
        # nodes may be QIDs or literal datatype endpoints.
        if not is_qid(source):
            continue
        if not (is_qid(target) or is_literal_endpoint(target)):
            continue

        instantiated.append(
            Edge(source=source, pid=edge.pid, target=target, edge_type=edge.edge_type)
        )

    return instantiated


def find_paths_between_nodes(
    edges: List[Edge],
    start_node: str,
    target_node: str,
    max_edges: int = MAX_PATH_EDGES,
) -> List[List[Edge]]:
    """Find simple directed paths in the instantiated SPARQL edge graph."""
    adjacency: Dict[str, List[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source].append(edge)

    found_paths: List[List[Edge]] = []
    queue = deque([(start_node, [], {start_node})])

    while queue:
        current_node, path_edges, visited_nodes = queue.popleft()
        if len(path_edges) >= max_edges:
            continue

        for edge in adjacency.get(current_node, []):
            if edge.target in visited_nodes and edge.target != target_node:
                continue

            new_path_edges = path_edges + [edge]
            if edge.target == target_node:
                found_paths.append(new_path_edges)
                continue

            # Literal endpoints cannot have outgoing Wikidata edges.
            if is_literal_endpoint(edge.target):
                continue

            queue.append((edge.target, new_path_edges, visited_nodes | {edge.target}))

    return found_paths


def edge_path_to_tokens(path_edges: List[Edge]) -> Tuple[str, ...]:
    """Convert edges into ordered tokens: Q1, P1, Q2, P2, Q3."""
    if not path_edges:
        return tuple()
    tokens: List[str] = [path_edges[0].source]
    for edge in path_edges:
        tokens.append(edge.pid)
        tokens.append(edge.target)
    return tuple(tokens)


def choose_candidate_pairs(
    sparql: str,
    var_bindings: Dict[str, Binding],
) -> List[Tuple[str, str]]:
    """
    Choose candidate endpoints to try connecting.

    Preferred pattern:
        fixed wd:Q anchors in the SPARQL connected to selected result variables.

    Literal-valued variables are included as possible Entity_2 endpoints, but only
    in the fixed-QID -> literal direction.
    """
    fixed_qids = extract_fixed_qids(sparql)
    select_vars = extract_select_variables(sparql)

    selected_qids: List[str] = []
    selected_literals: List[str] = []

    for var in select_vars:
        binding = var_bindings.get(var)
        if not binding:
            continue

        # Do not ignore variables ending in Label if they actually contain QIDs.
        if binding.kind == "qid":
            selected_qids.append(binding.value)
        elif binding.kind == "literal":
            selected_literals.append(binding.value)

    selected_qids = unique_preserve_order(selected_qids)
    selected_literals = unique_preserve_order(selected_literals)

    pairs: List[Tuple[str, str]] = []

    for fixed_qid in fixed_qids:
        for selected_qid in selected_qids:
            if fixed_qid == selected_qid:
                continue
            pairs.append((fixed_qid, selected_qid))
            pairs.append((selected_qid, fixed_qid))

        for literal_endpoint in selected_literals:
            pairs.append((fixed_qid, literal_endpoint))

    # If there are no fixed anchors, try pairs among selected QIDs.
    if not pairs and len(selected_qids) >= 2:
        for i, left in enumerate(selected_qids):
            for right in selected_qids[i + 1:]:
                if left == right:
                    continue
                pairs.append((left, right))
                pairs.append((right, left))

    seen: Set[Tuple[str, str]] = set()
    output: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            output.append(pair)
    return output


def derive_paths_for_result_row(
    source_row: Dict[str, str],
    parsed_result_row: Dict[str, Dict[str, str]],
) -> Tuple[List[PathResult], Dict[str, str]]:
    """Derive concrete paths for one parsed result row."""
    sparql = source_row.get("sparql", "")
    question = source_row.get("question", "")
    gold_answer = source_row.get("gold_answer", "")

    var_bindings, row_labels = build_variable_bindings_and_row_labels(parsed_result_row)
    symbolic_edges = parse_sparql_edges(sparql)
    instantiated_edges = instantiate_edges(symbolic_edges, var_bindings)

    if not instantiated_edges:
        return [], row_labels

    candidate_pairs = choose_candidate_pairs(sparql, var_bindings)
    results: List[PathResult] = []

    for entity_1, entity_2 in candidate_pairs:
        paths = find_paths_between_nodes(
            instantiated_edges,
            start_node=entity_1,
            target_node=entity_2,
            max_edges=MAX_PATH_EDGES,
        )

        for path_edges in paths:
            property_ids = tuple(edge.pid for edge in path_edges if edge.edge_type == "property")
            qualifier_ids = tuple(edge.pid for edge in path_edges if edge.edge_type == "qualifier")
            path_tokens = edge_path_to_tokens(path_edges)

            results.append(
                PathResult(
                    question=question,
                    gold_answer=gold_answer,
                    entity_1=entity_1,
                    entity_2=entity_2,
                    path_tokens=path_tokens,
                    property_ids=property_ids,
                    qualifier_ids=qualifier_ids,
                )
            )

    return deduplicate_path_results(results), row_labels


def deduplicate_path_results(results: List[PathResult]) -> List[PathResult]:
    seen: Set[Tuple[str, str, str, str, Tuple[str, ...]]] = set()
    output: List[PathResult] = []
    for result in results:
        key = (
            result.question,
            result.gold_answer,
            result.entity_1,
            result.entity_2,
            result.path_tokens,
        )
        if key not in seen:
            seen.add(key)
            output.append(result)
    return output


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def collect_ids_needed_for_labels(path_results: Iterable[PathResult]) -> List[str]:
    ids: List[str] = []
    for result in path_results:
        if is_wikidata_id(result.entity_1):
            ids.append(result.entity_1)
        if is_wikidata_id(result.entity_2):
            ids.append(result.entity_2)
        ids.extend(token for token in result.path_tokens if is_wikidata_id(token))
        ids.extend(result.property_ids)
        ids.extend(result.qualifier_ids)
    return unique_preserve_order(ids)


def format_path_tokens(
    path_tokens: Tuple[str, ...],
    row_labels: Dict[str, str],
    label_store: LabelStore,
) -> str:
    return "->".join(label_store.readable_id(token, row_labels) for token in path_tokens)


def group_path_results_by_pair(
    path_results: List[PathResult],
) -> Dict[Tuple[str, str, str, str], List[PathResult]]:
    grouped: Dict[Tuple[str, str, str, str], List[PathResult]] = defaultdict(list)
    for result in path_results:
        grouped[(result.question, result.gold_answer, result.entity_1, result.entity_2)].append(result)
    return grouped


def summarize_grouped_paths(
    grouped_paths: List[PathResult],
    row_labels: Dict[str, str],
    label_store: LabelStore,
) -> Dict[str, str]:
    """Create one output CSV row for one question/gold/entity pair."""
    if not grouped_paths:
        return {}

    first = grouped_paths[0]
    path_strings = [format_path_tokens(result.path_tokens, row_labels, label_store) for result in grouped_paths]
    path_strings = unique_preserve_order(path_strings)

    if len(path_strings) == 1:
        connection_path = path_strings[0]
    else:
        connection_path = "\n".join(f"Path{i + 1}: {path}" for i, path in enumerate(path_strings))

    property_ids: List[str] = []
    qualifier_ids: List[str] = []
    for result in grouped_paths:
        property_ids.extend(result.property_ids)
        qualifier_ids.extend(result.qualifier_ids)

    property_ids = unique_preserve_order(property_ids)
    qualifier_ids = unique_preserve_order(qualifier_ids)

    property_list = ";".join(label_store.readable_id(pid, row_labels) for pid in property_ids)
    qualifier_list = ";".join(label_store.readable_id(pid, row_labels) for pid in qualifier_ids)

    return {
        "question": first.question,
        "gold_answer": first.gold_answer,
        "Entity_1": label_store.readable_id(first.entity_1, row_labels),
        "Entity_2": label_store.readable_id(first.entity_2, row_labels),
        "Connection_Path": connection_path,
        "Property_Number": str(len(property_ids)) if property_ids else "",
        "Property_list": property_list,
        "Qualifier_Number": str(len(qualifier_ids)) if qualifier_ids else "",
        "Qualifier_list": qualifier_list,
    }


# ============================================================
# DIAGNOSTICS
# ============================================================

def diagnose_unresolved_source_row(
    source_row: Dict[str, str],
    parsed_result_rows: List[Dict[str, Dict[str, str]]],
) -> str:
    """Return a compact reason when a source row produced no paths."""
    sparql = source_row.get("sparql", "")

    if not str(source_row.get("result", "")).strip():
        return "empty result cell"
    if not parsed_result_rows:
        return "could not parse result as a Markdown table"

    symbolic_edges = parse_sparql_edges(sparql)
    if not symbolic_edges:
        return (
            "no supported SPARQL edges parsed; query may use unsupported syntax "
            "such as property paths, UNION-only patterns, VALUES, BIND, SERVICE-only triples, or aggregation"
        )

    any_instantiated = False
    any_candidates = False
    any_literal_binding = False

    for parsed_result_row in parsed_result_rows:
        var_bindings, _ = build_variable_bindings_and_row_labels(parsed_result_row)
        if any(binding.kind == "literal" for binding in var_bindings.values()):
            any_literal_binding = True
        instantiated = instantiate_edges(symbolic_edges, var_bindings)
        candidates = choose_candidate_pairs(sparql, var_bindings)
        if instantiated:
            any_instantiated = True
        if candidates:
            any_candidates = True
        for e1, e2 in candidates:
            if find_paths_between_nodes(instantiated, e1, e2):
                return "unexpected: diagnostic found a path"

    if not any_instantiated:
        if any_literal_binding:
            return "variables include literal values, but parsed SPARQL edges could not be filled into a supported path"
        return "SPARQL edges were parsed, but variables could not be filled from the result table"
    if not any_candidates:
        return "no candidate Entity_1/Entity_2 pair could be inferred from fixed wd:Q anchors and selected result variables"
    return "edges and candidate pairs exist, but no directed path connects them"


# ============================================================
# CSV PROCESSING
# ============================================================

def read_source_rows(input_csv_path: str) -> List[Dict[str, str]]:
    """Read source CSV rows into memory."""
    with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        required = {"result", "sparql"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        # question/gold_answer are optional for flexibility, but the output will
        # include blank cells if they are missing.
        return list(reader)


def derive_all_path_groups(
    source_rows: List[Dict[str, str]],
    label_store: LabelStore,
) -> Tuple[List[Tuple[List[PathResult], Dict[str, str]]], List[Dict[str, str]]]:
    """
    First pass over the input.

    Returns:
        all_groups:
            grouped PathResult objects plus row-specific labels.

        skipped_rows:
            diagnostic rows for source rows that produced no output path.
    """
    all_groups: List[Tuple[List[PathResult], Dict[str, str]]] = []
    skipped_rows: List[Dict[str, str]] = []

    for source_index, source_row in enumerate(
        tqdm(source_rows, desc="Deriving paths", unit="source row"),
        start=1,
    ):
        result_text = source_row.get("result", "")
        parsed_result_rows = parse_markdown_result_table(result_text)
        source_row_produced_any_path = False

        for result_index, parsed_result_row in enumerate(parsed_result_rows, start=1):
            path_results, row_labels = derive_paths_for_result_row(source_row, parsed_result_row)
            label_store.add_labels(row_labels)

            if not path_results:
                continue

            source_row_produced_any_path = True
            grouped = group_path_results_by_pair(path_results)
            for grouped_paths in grouped.values():
                all_groups.append((grouped_paths, row_labels))

        if not source_row_produced_any_path:
            reason = diagnose_unresolved_source_row(source_row, parsed_result_rows)
            skipped_rows.append(
                {
                    "source_row_number": str(source_index),
                    "question": source_row.get("question", ""),
                    "gold_answer": source_row.get("gold_answer", ""),
                    "reason": reason,
                    "sparql_preview": str(source_row.get("sparql", "")).replace("\n", " ")[:700],
                    "result_preview": str(result_text).replace("\n", " ")[:700],
                }
            )

            if WRITE_UNRESOLVED_ROWS:
                all_groups.append(([], {}))

    return all_groups, skipped_rows


def format_output_rows(
    all_groups: List[Tuple[List[PathResult], Dict[str, str]]],
    label_store: LabelStore,
) -> List[Dict[str, str]]:
    """Format final output rows after labels have been collected/fetched."""
    output_rows: List[Dict[str, str]] = []

    for grouped_paths, row_labels in tqdm(all_groups, desc="Formatting output", unit="row"):
        output_row = summarize_grouped_paths(grouped_paths, row_labels, label_store)
        if output_row:
            output_rows.append(output_row)

    return output_rows


def write_output_csv(output_csv_path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "question",
        "gold_answer",
        "Entity_1",
        "Entity_2",
        "Connection_Path",
        "Property_Number",
        "Property_list",
        "Qualifier_Number",
        "Qualifier_list",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_skipped_report_csv(output_csv_path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
        "source_row_number",
        "question",
        "gold_answer",
        "reason",
        "sparql_preview",
        "result_preview",
    ]

    with open(output_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        label_store = LabelStore(language=LABEL_LANGUAGE)
        source_rows = read_source_rows(INPUT_CSV_PATH)

        all_groups, skipped_rows = derive_all_path_groups(
            source_rows=source_rows,
            label_store=label_store,
        )

        # Collect all QIDs/PIDs that will appear in output, then fetch missing
        # labels in batches. Literal endpoints such as xsd:decimal are not sent
        # to Wikidata.
        ids_needed: List[str] = []
        for grouped_paths, _row_labels in all_groups:
            ids_needed.extend(collect_ids_needed_for_labels(grouped_paths))
        label_store.fetch_missing_labels(ids_needed)

        output_rows = format_output_rows(all_groups, label_store)

        write_output_csv(OUTPUT_CSV_PATH, output_rows)
        write_skipped_report_csv(SKIPPED_REPORT_CSV_PATH, skipped_rows)

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Input read from: {INPUT_CSV_PATH}")
    print(f"Output written to: {OUTPUT_CSV_PATH}")
    print(f"Rows written: {len(output_rows)}")
    print(f"Skipped source rows: {len(skipped_rows)}")
    print(f"Skipped report written to: {SKIPPED_REPORT_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())