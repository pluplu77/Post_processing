#!/usr/bin/env python3
"""
derive_connection_paths_from_sparql_results.py

Purpose
-------
This script reads an input CSV that contains two important columns:

    result
    sparql

Each row represents a SPARQL query and the query result table. The script derives
Wikidata connection paths directly from those two columns and writes a new CSV
with this schema:

    Entity_1,
    Entity_2,
    Connection_Path,
    Property_Number,
    Property_list,
    Qualifier_Number,
    Qualifier_list

Label strategy
--------------
The script prioritizes labels already present in the input CSV:

1. First, it reads labels from the `result` Markdown table.
   Example cell:

       Hamilton (wd:Q84323848)

   gives:

       Q84323848 -> Hamilton

   Example label cell:

       Hamilton (lang:en)

   can also be matched to the base variable if the column is named entityLabel.

2. If a QID, PID, or qualifier PID does not have a label in the CSV, the script
   falls back to the Wikidata API:

       https://www.wikidata.org/w/api.php?action=wbgetentities

Path strategy
-------------
The script does NOT use Wikidata to discover paths. It discovers paths from the
SPARQL text itself.

For example, this SPARQL triple:

    wd:Q1646482 wdt:P800 ?entity .

means:

    Q1646482 -> P800 -> ?entity

Then the result table tells us the concrete value of ?entity, such as:

    Hamilton (wd:Q84323848)

So the final readable path can be:

    Lin-Manuel Miranda (Q1646482)->notable work (P800)->Hamilton (Q84323848)

Supported SPARQL patterns
-------------------------
This script supports common Wikidata patterns:

1. Direct truthy properties:

       wd:Q1 wdt:P123 ?x .
       ?x wdt:P456 wd:Q2 .
       ?x wdt:P789 ?y .

2. Statement/qualifier paths:

       wd:Q1 p:P1346 ?statement .
       ?statement ps:P1346 ?winner .
       ?statement pq:P1686 ?work .

   This becomes:

       Q1 -> P1346 -> winner -> P1686 -> work

   where P1346 is counted as a property and P1686 is counted as a qualifier.

3. Multiple paths in one query. The output path is formatted as:

       Path1: ...
       Path2: ...

Limitations
-----------
The script is intentionally conservative. It derives paths only from explicit
SPARQL triple patterns in the `sparql` column. If the relationship is not
present in the SPARQL, the script cannot infer it from the result alone.

It does not fully parse every possible SPARQL feature, such as complex OPTIONAL,
UNION semantics, property paths like wdt:P31/wdt:P279*, SERVICE blocks, BINDed
IRIs, VALUES tables, or nested subqueries. It handles the common direct and
qualifier patterns used in many Wikidata QA datasets.

How to run
----------
1. Install dependencies:

       pip install requests tqdm

2. Set INPUT_CSV_PATH below.

3. Run:

       python derive_connection_paths_from_sparql_results.py
"""

from __future__ import annotations

import csv
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

# Change these paths as needed.
INPUT_CSV_PATH = "all_valid_cases.csv"
OUTPUT_CSV_PATH = "derived_connection_output.csv"

LABEL_LANGUAGE = "en"

# If True, missing QID/PID labels are fetched from Wikidata.
# The script still prioritizes labels already present in the input CSV.
USE_WIKIDATA_API_FOR_MISSING_LABELS = True

# Batch size for wbgetentities. Wikidata allows many IDs in one request, but
# keeping the batch moderate is friendlier to the public API.
WIKIDATA_LABEL_BATCH_SIZE = 40

# Sleep between API batches. Increase this if you process very large files.
WIKIDATA_API_SLEEP_SECONDS = 0.05

WBGETENTITIES_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "User-Agent": (
        "SPARQLResultConnectionPathDeriver/2.0 "
        "(Python requests; contact: your-email@example.com)"
    )
}

# Maximum number of graph edges to traverse when connecting instantiated SPARQL
# triples. Most generated QA paths are short, usually 1-3 edges.
MAX_PATH_EDGES = 5


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class Edge:
    """
    One directed edge parsed from the SPARQL.

    source and target may initially be either:
        - QIDs, such as Q1646482
        - variable names without '?', such as entity

    pid is always a property ID, such as P800.

    edge_type:
        "property"  = normal direct/main property
        "qualifier" = qualifier property, usually from pq:Pxxx
    """

    source: str
    pid: str
    target: str
    edge_type: str


@dataclass(frozen=True)
class PathResult:
    """
    One concrete path after variables have been filled with QIDs from a result row.
    """

    entity_1: str
    entity_2: str
    path_tokens: Tuple[str, ...]
    property_ids: Tuple[str, ...]
    qualifier_ids: Tuple[str, ...]


# ============================================================
# GENERIC HELPERS
# ============================================================

def unique_preserve_order(values: Iterable[str]) -> List[str]:
    """Return unique values in first-seen order."""
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
    token = str(token or "").strip()

    # Remove trailing punctuation sometimes caught by loose parsing.
    token = token.rstrip(".;,")

    # Full IRI forms.
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
    """Remove line comments from SPARQL."""
    lines = []
    for line in str(sparql or "").splitlines():
        # Remove comments that start with #. This is a simple heuristic and does
        # not try to handle # inside string literals.
        lines.append(re.sub(r"#.*$", "", line))
    return "\n".join(lines)


# ============================================================
# LABEL STORE
# ============================================================

class LabelStore:
    """
    Stores labels found in the input CSV and fetches missing labels from Wikidata.

    Priority:
        1. row-level labels from the `result` table
        2. global labels already discovered from other rows
        3. Wikidata API fallback
        4. raw ID if no label is available
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

        if not (is_qid(entity_id) or is_pid(entity_id)):
            return

        # Do not overwrite a good label with the raw ID.
        if label == entity_id:
            return

        self.global_labels.setdefault(entity_id, label)

    def add_labels(self, labels: Dict[str, str]) -> None:
        for entity_id, label in labels.items():
            self.add_label(entity_id, label)

    def fetch_missing_labels(self, ids: Iterable[str]) -> None:
        """
        Fetch missing QID/PID labels in batches.

        This is called after collecting the IDs needed for the output. It avoids
        doing one HTTP request per token.
        """
        if not USE_WIKIDATA_API_FOR_MISSING_LABELS:
            return

        ids_to_fetch = [
            item_id
            for item_id in unique_preserve_order(ids)
            if (is_qid(item_id) or is_pid(item_id))
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
                    lang_label = labels.get(self.language, {})
                    value = lang_label.get("value", "")

                    if value:
                        self.global_labels[item_id] = value
                    else:
                        self.missing_cache.add(item_id)

            except Exception as exc:
                print(
                    f"Warning: failed to fetch labels for batch {batch}: {exc}",
                    file=sys.stderr,
                )
                self.missing_cache.update(batch)

            time.sleep(WIKIDATA_API_SLEEP_SECONDS)

    def label_for(self, entity_id: str, row_labels: Optional[Dict[str, str]] = None) -> str:
        entity_id = str(entity_id or "").strip().upper()

        if row_labels and entity_id in row_labels:
            return row_labels[entity_id]

        if entity_id in self.global_labels:
            return self.global_labels[entity_id]

        return entity_id

    def readable_id(self, entity_id: str, row_labels: Optional[Dict[str, str]] = None) -> str:
        entity_id = str(entity_id or "").strip().upper()
        label = self.label_for(entity_id, row_labels)
        return f"{label} ({entity_id})"


# ============================================================
# RESULT TABLE PARSING
# ============================================================

def split_markdown_row(line: str) -> List[str]:
    """
    Split a simple Markdown table row.

    This assumes the cells themselves do not contain unescaped vertical bars.
    That matches the common format in the provided CSV.
    """
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
        for work (wd:P1686)
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

    # Literal label cell such as Hamilton (lang:en).
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

    parsed["label"] = cell
    return parsed


def parse_markdown_result_table(result_text: str) -> List[Dict[str, Dict[str, str]]]:
    """
    Parse a Markdown result table from the `result` column.

    Returns a list of result rows. Each row is a mapping:

        column name -> parsed cell dict

    Example return:

        [
            {
                "entity": {
                    "qid": "Q84323848",
                    "label": "Hamilton",
                    ...
                },
                "label": {
                    "label": "Hamilton",
                    ...
                }
            }
        ]
    """
    text = str(result_text or "").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    table_lines = [line for line in lines if line.startswith("|") and line.endswith("|")]
    if len(table_lines) < 2:
        return []

    headers = split_markdown_row(table_lines[0])
    rows: List[Dict[str, Dict[str, str]]] = []

    for line in table_lines[1:]:
        # Skip separator row like | ----- | ----- |
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
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build variable bindings and labels from one parsed result row.

    Returns:
        var_bindings:
            SPARQL variable name -> concrete QID/PID

        row_labels:
            QID/PID -> label from this row

    Notes:
        - Variables are represented without the leading '?'.
        - QID-valued and PID-valued result cells are both bound.
        - Columns named somethingLabel are used as labels for something.
    """
    var_bindings: Dict[str, str] = {}
    row_labels: Dict[str, str] = {}

    # First pass: bind direct entity/property columns.
    for column, cell in parsed_result_row.items():
        item_id = cell.get("id") or cell.get("qid") or cell.get("pid") or ""
        label = cell.get("label", "")

        if item_id:
            var_bindings[column] = item_id
            if label and label != item_id:
                row_labels[item_id] = label

    # Second pass: match label columns to base columns.
    # Common patterns:
    #     entity + entityLabel
    #     entity + label
    for column, cell in parsed_result_row.items():
        label = cell.get("label", "")
        if not label:
            continue

        possible_base_columns: List[str] = []

        if column.endswith("Label"):
            possible_base_columns.append(column[:-5])

        if column.lower() == "label":
            # If there is exactly one QID-valued variable, the generic ?label
            # column probably labels that variable.
            qid_columns = [
                name for name, value in var_bindings.items()
                if is_qid(value)
            ]
            if len(qid_columns) == 1:
                possible_base_columns.append(qid_columns[0])

        for base_column in possible_base_columns:
            bound_id = var_bindings.get(base_column)
            if bound_id:
                row_labels[bound_id] = label

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

    Supported examples:
        wd:Q1 wdt:P123 ?entity .
        ?entity wdt:P456 wd:Q2 .
        ?a wdt:P31 ?b .
    """
    query = strip_sparql_comments(sparql)

    token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    pattern = re.compile(
        rf"(?P<s>{token})\s+wdt:(?P<p>P[1-9]\d*)\s+(?P<o>{token})\s*\." ,
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

    The first edge is edge_type='property'.
    The second edge is edge_type='qualifier'.
    """
    query = strip_sparql_comments(sparql)

    subject_token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    object_token = r"(?:wd:Q[1-9]\d*|\?[A-Za-z_][A-Za-z0-9_]*)"
    statement_token = r"\?[A-Za-z_][A-Za-z0-9_]*"

    claim_pattern = re.compile(
        rf"(?P<subject>{subject_token})\s+"
        rf"p:(?P<mainprop>P[1-9]\d*)\s+"
        rf"(?P<statement>{statement_token})\s*\.",
        flags=re.IGNORECASE,
    )

    ps_pattern = re.compile(
        rf"(?P<statement>{statement_token})\s+"
        rf"ps:(?P<mainprop>P[1-9]\d*)\s+"
        rf"(?P<mainvalue>{object_token})\s*\.",
        flags=re.IGNORECASE,
    )

    pq_pattern = re.compile(
        rf"(?P<statement>{statement_token})\s+"
        rf"pq:(?P<qualprop>P[1-9]\d*)\s+"
        rf"(?P<qualvalue>{object_token})\s*\.",
        flags=re.IGNORECASE,
    )

    # statement variable -> (subject, main property)
    claim_by_statement: Dict[str, Tuple[str, str]] = {}

    # statement variable -> (main property, main value)
    main_value_by_statement: Dict[str, Tuple[str, str]] = {}

    # statement variable -> list of (qualifier property, qualifier value)
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

        edges.append(
            Edge(
                source=subject,
                pid=claim_mainprop,
                target=mainvalue,
                edge_type="property",
            )
        )

        for qualprop, qualvalue in qualifiers_by_statement.get(statement, []):
            edges.append(
                Edge(
                    source=mainvalue,
                    pid=qualprop,
                    target=qualvalue,
                    edge_type="qualifier",
                )
            )

    return edges


def parse_sparql_edges(sparql: str) -> List[Edge]:
    """Parse all supported edge types from SPARQL."""
    edges: List[Edge] = []
    edges.extend(parse_direct_wdt_edges(sparql))
    edges.extend(parse_qualifier_edges(sparql))

    # De-duplicate while preserving order.
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

def instantiate_token(token: str, var_bindings: Dict[str, str]) -> Optional[str]:
    """
    Convert a symbolic token into a concrete QID/PID if possible.

    QIDs and PIDs are already concrete. Variable names are looked up in the
    result-row bindings.
    """
    token = str(token or "").strip()

    if is_qid(token) or is_pid(token):
        return token

    return var_bindings.get(token)


def instantiate_edges(edges: List[Edge], var_bindings: Dict[str, str]) -> List[Edge]:
    """Fill SPARQL variables in edges using the current result row."""
    instantiated: List[Edge] = []

    for edge in edges:
        source = instantiate_token(edge.source, var_bindings)
        target = instantiate_token(edge.target, var_bindings)

        # For our output, path nodes must be concrete QIDs. The predicate must be
        # a concrete PID.
        if not source or not target:
            continue
        if not is_qid(source) or not is_qid(target):
            continue
        if not is_pid(edge.pid):
            continue

        instantiated.append(
            Edge(
                source=source,
                pid=edge.pid,
                target=target,
                edge_type=edge.edge_type,
            )
        )

    return instantiated


def find_paths_between_entities(
    edges: List[Edge],
    start_qid: str,
    target_qid: str,
    max_edges: int = MAX_PATH_EDGES,
) -> List[List[Edge]]:
    """Find simple directed paths in the instantiated SPARQL edge graph."""
    adjacency: Dict[str, List[Edge]] = defaultdict(list)
    for edge in edges:
        adjacency[edge.source].append(edge)

    found_paths: List[List[Edge]] = []
    queue = deque([(start_qid, [], {start_qid})])

    while queue:
        current_qid, path_edges, visited_nodes = queue.popleft()

        if len(path_edges) >= max_edges:
            continue

        for edge in adjacency.get(current_qid, []):
            if edge.target in visited_nodes and edge.target != target_qid:
                continue

            new_path_edges = path_edges + [edge]

            if edge.target == target_qid:
                found_paths.append(new_path_edges)
                continue

            queue.append(
                (
                    edge.target,
                    new_path_edges,
                    visited_nodes | {edge.target},
                )
            )

    return found_paths


def edge_path_to_tokens(path_edges: List[Edge]) -> Tuple[str, ...]:
    """
    Convert edges into ordered tokens:

        [Q1, P1, Q2, P2, Q3]
    """
    if not path_edges:
        return tuple()

    tokens: List[str] = [path_edges[0].source]
    for edge in path_edges:
        tokens.append(edge.pid)
        tokens.append(edge.target)

    return tuple(tokens)


def choose_candidate_entity_pairs(
    sparql: str,
    var_bindings: Dict[str, str],
) -> List[Tuple[str, str]]:
    """
    Choose candidate entity pairs to try connecting.

    Preferred pattern:
        fixed wd:Q anchors in the SPARQL connected to selected result variables.

    Example:
        SELECT ?entity ?label WHERE {
          wd:Q1646482 wdt:P800 ?entity .
        }

    Candidate:
        Q1646482 -> bound value of ?entity

    Also tries the reverse direction because some queries are written as:

        ?film wdt:P170 wd:Q363402 .

    In that case, the actual path is selected film -> fixed Q363402.
    """
    fixed_qids = extract_fixed_qids(sparql)
    select_vars = extract_select_variables(sparql)

    selected_qids: List[str] = []
    for var in select_vars:
        # Do not treat label variables as entities.
        if var.lower().endswith("label"):
            continue

        bound = var_bindings.get(var)
        if bound and is_qid(bound):
            selected_qids.append(bound)

    selected_qids = unique_preserve_order(selected_qids)

    pairs: List[Tuple[str, str]] = []

    for fixed_qid in fixed_qids:
        for selected_qid in selected_qids:
            if fixed_qid == selected_qid:
                continue
            pairs.append((fixed_qid, selected_qid))
            pairs.append((selected_qid, fixed_qid))

    # If there are no fixed anchors, try selected variable pairs.
    if not pairs and len(selected_qids) >= 2:
        for i, left in enumerate(selected_qids):
            for right in selected_qids[i + 1:]:
                if left == right:
                    continue
                pairs.append((left, right))
                pairs.append((right, left))

    # De-duplicate pairs.
    seen: Set[Tuple[str, str]] = set()
    output: List[Tuple[str, str]] = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            output.append(pair)

    return output


def derive_paths_for_result_row(
    sparql: str,
    parsed_result_row: Dict[str, Dict[str, str]],
) -> Tuple[List[PathResult], Dict[str, str]]:
    """
    Derive concrete paths for one SPARQL result row.

    Returns:
        path results
        row-level labels
    """
    var_bindings, row_labels = build_variable_bindings_and_row_labels(parsed_result_row)

    symbolic_edges = parse_sparql_edges(sparql)
    instantiated_edges = instantiate_edges(symbolic_edges, var_bindings)

    if not instantiated_edges:
        return [], row_labels

    candidate_pairs = choose_candidate_entity_pairs(sparql, var_bindings)

    results: List[PathResult] = []

    for entity_1, entity_2 in candidate_pairs:
        paths = find_paths_between_entities(
            instantiated_edges,
            start_qid=entity_1,
            target_qid=entity_2,
            max_edges=MAX_PATH_EDGES,
        )

        for path_edges in paths:
            property_ids = tuple(
                edge.pid for edge in path_edges if edge.edge_type == "property"
            )
            qualifier_ids = tuple(
                edge.pid for edge in path_edges if edge.edge_type == "qualifier"
            )
            path_tokens = edge_path_to_tokens(path_edges)

            results.append(
                PathResult(
                    entity_1=entity_1,
                    entity_2=entity_2,
                    path_tokens=path_tokens,
                    property_ids=property_ids,
                    qualifier_ids=qualifier_ids,
                )
            )

    return deduplicate_path_results(results), row_labels


def deduplicate_path_results(results: List[PathResult]) -> List[PathResult]:
    seen: Set[Tuple[str, str, Tuple[str, ...]]] = set()
    output: List[PathResult] = []

    for result in results:
        key = (result.entity_1, result.entity_2, result.path_tokens)
        if key not in seen:
            seen.add(key)
            output.append(result)

    return output


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def collect_ids_needed_for_labels(path_results: Iterable[PathResult]) -> List[str]:
    """Collect all QIDs/PIDs that will appear in output."""
    ids: List[str] = []

    for result in path_results:
        ids.append(result.entity_1)
        ids.append(result.entity_2)
        ids.extend(result.path_tokens)
        ids.extend(result.property_ids)
        ids.extend(result.qualifier_ids)

    return [item_id for item_id in unique_preserve_order(ids) if is_qid(item_id) or is_pid(item_id)]


def format_path_tokens(
    path_tokens: Tuple[str, ...],
    row_labels: Dict[str, str],
    label_store: LabelStore,
) -> str:
    readable_tokens: List[str] = []

    for token in path_tokens:
        if is_qid(token) or is_pid(token):
            readable_tokens.append(label_store.readable_id(token, row_labels))
        else:
            readable_tokens.append(token)

    return "->".join(readable_tokens)


def group_path_results_by_pair(
    path_results: List[PathResult],
) -> Dict[Tuple[str, str], List[PathResult]]:
    grouped: Dict[Tuple[str, str], List[PathResult]] = defaultdict(list)

    for result in path_results:
        grouped[(result.entity_1, result.entity_2)].append(result)

    return grouped


def summarize_grouped_paths(
    grouped_paths: List[PathResult],
    row_labels: Dict[str, str],
    label_store: LabelStore,
) -> Dict[str, str]:
    """Create one output CSV row for one Entity_1/Entity_2 pair."""
    if not grouped_paths:
        return {}

    entity_1 = grouped_paths[0].entity_1
    entity_2 = grouped_paths[0].entity_2

    path_strings = [
        format_path_tokens(result.path_tokens, row_labels, label_store)
        for result in grouped_paths
    ]
    path_strings = unique_preserve_order(path_strings)

    if len(path_strings) == 1:
        connection_path = path_strings[0]
    else:
        connection_path = "\n".join(
            f"Path{i + 1}: {path}"
            for i, path in enumerate(path_strings)
        )

    property_ids: List[str] = []
    qualifier_ids: List[str] = []

    for result in grouped_paths:
        property_ids.extend(result.property_ids)
        qualifier_ids.extend(result.qualifier_ids)

    property_ids = unique_preserve_order(property_ids)
    qualifier_ids = unique_preserve_order(qualifier_ids)

    property_list = ";".join(
        label_store.readable_id(pid, row_labels) for pid in property_ids
    )
    qualifier_list = ";".join(
        label_store.readable_id(pid, row_labels) for pid in qualifier_ids
    )

    return {
        "Entity_1": label_store.readable_id(entity_1, row_labels),
        "Entity_2": label_store.readable_id(entity_2, row_labels),
        "Connection_Path": connection_path,
        "Property_Number": str(len(property_ids)) if property_ids else "",
        "Property_list": property_list,
        "Qualifier_Number": str(len(qualifier_ids)) if qualifier_ids else "",
        "Qualifier_list": qualifier_list,
    }


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

        return list(reader)


def derive_all_path_groups(
    source_rows: List[Dict[str, str]],
    label_store: LabelStore,
) -> List[Tuple[List[PathResult], Dict[str, str]]]:
    """
    First pass over the data.

    It derives paths and collects labels that are already present in the result
    tables. It does not format the final output yet, because we want to batch
    fetch all missing labels first.
    """
    all_groups: List[Tuple[List[PathResult], Dict[str, str]]] = []

    for source_row in tqdm(source_rows, desc="Deriving paths", unit="source row"):
        result_text = source_row.get("result", "")
        sparql = source_row.get("sparql", "")

        parsed_result_rows = parse_markdown_result_table(result_text)

        for parsed_result_row in parsed_result_rows:
            path_results, row_labels = derive_paths_for_result_row(
                sparql=sparql,
                parsed_result_row=parsed_result_row,
            )

            # Save row labels globally so future rows can reuse them.
            label_store.add_labels(row_labels)

            if not path_results:
                continue

            grouped = group_path_results_by_pair(path_results)
            for grouped_paths in grouped.values():
                all_groups.append((grouped_paths, row_labels))

    return all_groups


def format_output_rows(
    all_groups: List[Tuple[List[PathResult], Dict[str, str]]],
    label_store: LabelStore,
) -> List[Dict[str, str]]:
    """Format final output rows after labels have been collected/fetched."""
    output_rows: List[Dict[str, str]] = []

    for grouped_paths, row_labels in tqdm(all_groups, desc="Formatting output", unit="row"):
        output_row = summarize_grouped_paths(
            grouped_paths=grouped_paths,
            row_labels=row_labels,
            label_store=label_store,
        )
        if output_row:
            output_rows.append(output_row)

    return output_rows


def write_output_csv(output_csv_path: str, rows: List[Dict[str, str]]) -> None:
    fieldnames = [
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


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        label_store = LabelStore(language=LABEL_LANGUAGE)

        source_rows = read_source_rows(INPUT_CSV_PATH)

        all_groups = derive_all_path_groups(
            source_rows=source_rows,
            label_store=label_store,
        )

        # Collect every ID that may appear in the output and fetch missing labels
        # in batches. Labels already found in the result table are not fetched.
        ids_needed: List[str] = []
        for grouped_paths, _row_labels in all_groups:
            ids_needed.extend(collect_ids_needed_for_labels(grouped_paths))

        label_store.fetch_missing_labels(ids_needed)

        output_rows = format_output_rows(
            all_groups=all_groups,
            label_store=label_store,
        )

        write_output_csv(
            OUTPUT_CSV_PATH,
            output_rows,
        )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Input read from: {INPUT_CSV_PATH}")
    print(f"Output written to: {OUTPUT_CSV_PATH}")
    print(f"Rows written: {len(output_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
