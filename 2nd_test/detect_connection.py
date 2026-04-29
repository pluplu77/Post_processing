#!/usr/bin/env python3

# ============================================================
# HOW TO RUN
# ============================================================
# 1. Install dependencies:
#       pip install requests tqdm
#
# 2. Edit the hardcoded file paths in the CONFIG section below.
#
# 3. Run:
#       python wikidata_entity1_first_paths.py
#
# ============================================================
# INPUT CSV
# ============================================================
# Required columns:
#   Entity_1
#   Entity_2
#
# Values may be raw QIDs:
#   Q503034,Q47221
#
# Or readable cells containing QIDs:
#   Los Angeles Film Critics Association Award for Best Actor (Q503034),Taxi Driver (Q47221)
#
# ============================================================
# OUTPUT CSV FORMAT
# ============================================================
# Entity_1,
# Entity_2,
# Connection_Path,
# Property_Number,
# Property_list,
# Qualifier_Number,
# Qualifier_list
#
# ============================================================
# IMPORTANT LOGIC
# ============================================================
# This version searches Entity_1's own Wikidata statement page first.
#
# Example:
#
#   Entity_1 = Q503034
#   Entity_2 = Q47221
#
# It reads:
#
#   https://www.wikidata.org/wiki/Special:EntityData/Q503034.json
#
# Then scans Q503034's claims.
#
# For the LAFCA Best Actor example, Wikidata stores this as:
#
#   Q503034
#     P1346 winner
#       main value: Q36949 Robert De Niro
#       qualifier: P1686 for work = Q47221 Taxi Driver
#
# So the preferred output path is:
#
#   Los Angeles Film Critics Association Award for Best Actor (Q503034)
#   ->Robert De Niro (Q36949)
#   ->for work (P1686)
#   ->Taxi Driver (Q47221)
#
# Property_list:
#   winner (P1346)
#
# Qualifier_list:
#   for work (P1686)
#
# This prevents a weaker fallback graph path like:
#
#   award -> winner -> Robert De Niro -> notable work -> Taxi Driver
#
# from being chosen when Entity_1's own statement qualifiers already explain
# the relationship.
# ============================================================

from __future__ import annotations

import csv
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

INPUT_CSV_PATH = "input.csv"
OUTPUT_CSV_PATH = "output.csv"
OUTPUT_LABEL_CSV_PATH = "output_label_name.csv"

LABEL_LANGUAGE = "en"

# Used only as a fallback when Entity_1's own page does not contain
# a direct statement or qualifier explanation.
MAX_FALLBACK_HOPS = 2

# Usually you do not want deprecated Wikidata statements.
INCLUDE_DEPRECATED_STATEMENTS = False

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
ENTITY_DATA_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"
WBGETENTITIES_API = "https://www.wikidata.org/w/api.php"

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "WikidataEntity1FirstPathFinder/3.0 "
                  "(Python requests; contact: your-email@example.com)",
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ConnectionResult:
    """
    property_ids:
        Main Wikidata statement properties from Entity_1.
        Example:
            ["P1346"]

    qualifier_ids:
        Qualifier properties on that statement.
        Example:
            ["P1686"]

    path_steps:
        Ordered display path.

        For direct Entity_1 statement:
            Entity_1 -> Entity_2

        For Entity_1 statement with qualifier:
            Entity_1 -> main statement value -> qualifier property -> Entity_2

        Example:
            [
                ("entity", "Q503034"),
                ("entity", "Q36949"),
                ("qualifier", "P1686"),
                ("entity", "Q47221"),
            ]

    source:
        Explanation source:
            "entity1_statement_direct"
            "entity1_statement_qualifier"
            "fallback_truthy_path"

    priority:
        Lower number = better.
    """
    property_ids: List[str]
    qualifier_ids: List[str]
    path_steps: List[Tuple[str, str]]
    source: str
    priority: int


# ============================================================
# BASIC PARSING / VALIDATION
# ============================================================

def extract_qid_from_cell(cell_value: str) -> str:
    text = str(cell_value).strip().upper()
    match = re.search(r"Q[1-9]\d*", text)
    if not match:
        raise ValueError(f"Could not find a valid QID in: {cell_value}")
    return match.group(0)


def validate_qid(qid: str) -> str:
    qid = str(qid).strip().upper()
    if not re.fullmatch(r"Q[1-9]\d*", qid):
        raise ValueError(f"Invalid QID: {qid}")
    return qid


def validate_pid(pid: str) -> str:
    pid = str(pid).strip().upper()
    if not re.fullmatch(r"P[1-9]\d*", pid):
        raise ValueError(f"Invalid PID: {pid}")
    return pid


def extract_last_path_segment(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def unique_preserve_order(values: List[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


# ============================================================
# NETWORK HELPERS
# ============================================================

def get_entity_json(qid: str, timeout: int = 30, retries: int = 2) -> Dict[str, Any]:
    url = ENTITY_DATA_URL.format(qid)
    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                url,
                headers=HEADERS,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"Could not fetch EntityData for {qid}: {exc}") from exc

    raise RuntimeError(f"Could not fetch EntityData for {qid}: {last_error}")


def entity_exists(qid: str) -> bool:
    try:
        data = get_entity_json(qid)
        return qid in data.get("entities", {})
    except Exception:
        return False


def run_sparql(query: str, timeout: int = 45, retries: int = 2) -> Dict[str, Any]:
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": HEADERS["User-Agent"],
    }

    last_error: Optional[Exception] = None

    for attempt in range(retries + 1):
        try:
            response = requests.get(
                SPARQL_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
            else:
                raise RuntimeError(f"SPARQL query failed: {exc}") from exc

    raise RuntimeError(f"SPARQL query failed: {last_error}")


# ============================================================
# LABEL HELPERS
# ============================================================

_label_cache: Dict[Tuple[str, str], str] = {}


def get_entity_or_property_label(entity_id: str, language: str = LABEL_LANGUAGE) -> str:
    cache_key = (entity_id, language)
    if cache_key in _label_cache:
        return _label_cache[cache_key]

    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "labels",
        "languages": language,
        "format": "json",
    }

    label = entity_id

    try:
        response = requests.get(
            WBGETENTITIES_API,
            params=params,
            headers={"User-Agent": HEADERS["User-Agent"]},
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()

        entity_data = data.get("entities", {}).get(entity_id, {})
        labels = entity_data.get("labels", {})
        if language in labels:
            label = labels[language]["value"]

    except Exception:
        label = entity_id

    _label_cache[cache_key] = label
    return label


def readable_entity(qid: str, language: str = LABEL_LANGUAGE) -> str:
    return f"{get_entity_or_property_label(qid, language)} ({qid})"


def readable_property(pid: str, language: str = LABEL_LANGUAGE) -> str:
    return f"{get_entity_or_property_label(pid, language)} ({pid})"


def readable_cell_entity(original_value: str, language: str = LABEL_LANGUAGE) -> str:
    try:
        qid = extract_qid_from_cell(original_value)
        return readable_entity(qid, language)
    except Exception:
        return str(original_value)


def readable_pid_list(raw_value: str, language: str = LABEL_LANGUAGE) -> str:
    raw_value = str(raw_value or "").strip()
    if not raw_value:
        return ""

    groups = [part.strip() for part in raw_value.split(";") if part.strip()]
    readable_groups: List[str] = []

    for group in groups:
        pids = re.findall(r"P[1-9]\d*", group.upper())
        if not pids:
            continue
        readable_groups.append(" / ".join(readable_property(pid, language) for pid in pids))

    return ";".join(readable_groups)


def build_path_string(
    path_steps: List[Tuple[str, str]],
    readable: bool,
    language: str = LABEL_LANGUAGE,
) -> str:
    parts: List[str] = []

    for kind, value in path_steps:
        if kind == "entity":
            parts.append(readable_entity(value, language) if readable else value)
        elif kind == "property":
            parts.append(readable_property(value, language) if readable else value)
        elif kind == "qualifier":
            parts.append(readable_property(value, language) if readable else value)
        else:
            parts.append(value)

    return "->".join(parts)


def readable_raw_connection_path(raw_path: str, language: str = LABEL_LANGUAGE) -> str:
    raw_path = str(raw_path or "").strip()
    if not raw_path:
        return ""

    readable_paths: List[str] = []

    for one_path in raw_path.split("\n"):
        tokens = [token.strip() for token in one_path.split("->") if token.strip()]
        readable_tokens: List[str] = []

        for token in tokens:
            if re.fullmatch(r"Q[1-9]\d*", token):
                readable_tokens.append(readable_entity(token, language))
            elif re.fullmatch(r"P[1-9]\d*", token):
                readable_tokens.append(readable_property(token, language))
            else:
                readable_tokens.append(token)

        readable_paths.append("->".join(readable_tokens))

    return "\n".join(readable_paths)


# ============================================================
# WIKIDATA JSON SNAK HELPERS
# ============================================================

def qid_from_snak(snak: Dict[str, Any]) -> Optional[str]:
    """
    Extract QID from a Wikidata snak if the snak value is a Wikidata item.

    Returns None for:
      - no value
      - some value with non-item datatype
      - literal values
    """
    if not isinstance(snak, dict):
        return None

    if snak.get("snaktype") != "value":
        return None

    datavalue = snak.get("datavalue")
    if not isinstance(datavalue, dict):
        return None

    value = datavalue.get("value")
    if not isinstance(value, dict):
        return None

    entity_type = value.get("entity-type")
    numeric_id = value.get("numeric-id")

    if entity_type != "item" or numeric_id is None:
        return None

    return f"Q{numeric_id}"


def get_claims_from_entity_json(entity_json: Dict[str, Any], qid: str) -> Dict[str, List[Dict[str, Any]]]:
    entity_data = entity_json.get("entities", {}).get(qid, {})
    claims = entity_data.get("claims", {})
    if not isinstance(claims, dict):
        return {}
    return claims


# ============================================================
# ENTITY_1 PAGE-FIRST SEARCH
# ============================================================

def find_connections_on_entity1_page(qid1: str, qid2: str) -> List[ConnectionResult]:
    """
    Search only inside Entity_1's own Wikidata EntityData JSON.

    This is the preferred search.

    It finds two important cases:

    Case A: direct main statement value

        Entity_1 --Pxxx--> Entity_2

        Example:
            Q503034 --P1346 winner--> Q36949

        Output path:
            Q503034->Q36949

        Property_list:
            P1346

        Qualifier_list:
            empty

    Case B: qualifier value on a statement

        Entity_1 --Pxxx--> main value
                    qualifier Pyyy --> Entity_2

        Example:
            Q503034 --P1346 winner--> Q36949
                    qualifier P1686 for work --> Q47221

        Output path:
            Q503034->Q36949->P1686->Q47221

        Property_list:
            P1346

        Qualifier_list:
            P1686
    """
    data = get_entity_json(qid1)
    claims = get_claims_from_entity_json(data, qid1)

    results: List[ConnectionResult] = []

    for property_id, statements in claims.items():
        if not re.fullmatch(r"P[1-9]\d*", property_id):
            continue

        if not isinstance(statements, list):
            continue

        for statement in statements:
            if not isinstance(statement, dict):
                continue

            rank = statement.get("rank")
            if rank == "deprecated" and not INCLUDE_DEPRECATED_STATEMENTS:
                continue

            mainsnak = statement.get("mainsnak", {})
            main_value_qid = qid_from_snak(mainsnak)

            # Case A:
            # Entity_1 has a direct statement whose value is Entity_2.
            if main_value_qid == qid2:
                results.append(
                    ConnectionResult(
                        property_ids=[property_id],
                        qualifier_ids=[],
                        path_steps=[
                            ("entity", qid1),
                            ("entity", qid2),
                        ],
                        source="entity1_statement_direct",
                        priority=1,
                    )
                )

            # Case B:
            # Entity_1 has a statement whose qualifier value is Entity_2.
            qualifiers = statement.get("qualifiers", {})
            if not isinstance(qualifiers, dict):
                continue

            for qualifier_id, qualifier_snaks in qualifiers.items():
                if not re.fullmatch(r"P[1-9]\d*", qualifier_id):
                    continue

                if not isinstance(qualifier_snaks, list):
                    continue

                for qualifier_snak in qualifier_snaks:
                    qualifier_value_qid = qid_from_snak(qualifier_snak)

                    if qualifier_value_qid != qid2:
                        continue

                    # The best explanatory path needs the main statement value.
                    # For the LAFCA example:
                    #   Q503034 -> Q36949 -> P1686 -> Q47221
                    #
                    # If the main value is not a QID, fall back to:
                    #   Q503034 -> P1686 -> Q47221
                    if main_value_qid:
                        path_steps = [
                            ("entity", qid1),
                            ("entity", main_value_qid),
                            ("qualifier", qualifier_id),
                            ("entity", qid2),
                        ]
                    else:
                        path_steps = [
                            ("entity", qid1),
                            ("qualifier", qualifier_id),
                            ("entity", qid2),
                        ]

                    results.append(
                        ConnectionResult(
                            property_ids=[property_id],
                            qualifier_ids=[qualifier_id],
                            path_steps=path_steps,
                            source="entity1_statement_qualifier",
                            priority=2,
                        )
                    )

    return deduplicate_connections(results)


# ============================================================
# FALLBACK GRAPH SEARCH
# ============================================================

def build_fallback_truthy_path_query(qid1: str, qid2: str, max_hops: int) -> str:
    """
    Fallback only.

    This searches general truthy Wikidata graph paths.

    It is intentionally lower priority than scanning Entity_1's own
    statements and qualifiers.
    """
    if max_hops < 1:
        raise ValueError("max_hops must be >= 1")

    union_blocks: List[str] = []

    # 1-hop:
    union_blocks.append(f"""
    {{
      wd:{qid1} ?p1 wd:{qid2} .
      FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
    }}
    """)

    # 2-hop:
    if max_hops >= 2:
        union_blocks.append(f"""
        {{
          wd:{qid1} ?p1 ?n1 .
          ?n1 ?p2 wd:{qid2} .

          FILTER(STRSTARTS(STR(?p1), "http://www.wikidata.org/prop/direct/"))
          FILTER(STRSTARTS(STR(?p2), "http://www.wikidata.org/prop/direct/"))
          FILTER(STRSTARTS(STR(?n1), "http://www.wikidata.org/entity/Q"))

          FILTER(?n1 != wd:{qid1})
          FILTER(?n1 != wd:{qid2})
        }}
        """)

    query = """
SELECT DISTINCT ?p1 ?n1 ?p2 WHERE {
"""
    query += "\nUNION\n".join(union_blocks)
    query += """
}
ORDER BY ?p1 ?p2 ?n1
"""
    return query


def find_fallback_truthy_paths(qid1: str, qid2: str, max_hops: int) -> List[ConnectionResult]:
    query = build_fallback_truthy_path_query(qid1, qid2, max_hops)
    data = run_sparql(query)
    bindings = data.get("results", {}).get("bindings", [])

    results: List[ConnectionResult] = []

    for row in bindings:
        p1 = extract_last_path_segment(row["p1"]["value"])

        # 1-hop fallback
        if "p2" not in row:
            results.append(
                ConnectionResult(
                    property_ids=[p1],
                    qualifier_ids=[],
                    path_steps=[
                        ("entity", qid1),
                        ("property", p1),
                        ("entity", qid2),
                    ],
                    source="fallback_truthy_path",
                    priority=100,
                )
            )
            continue

        # 2-hop fallback
        p2 = extract_last_path_segment(row["p2"]["value"])
        n1 = extract_last_path_segment(row["n1"]["value"])

        results.append(
            ConnectionResult(
                property_ids=[p1, p2],
                qualifier_ids=[],
                path_steps=[
                    ("entity", qid1),
                    ("property", p1),
                    ("entity", n1),
                    ("property", p2),
                    ("entity", qid2),
                ],
                source="fallback_truthy_path",
                priority=100,
            )
        )

    return deduplicate_connections(results)


# ============================================================
# CONNECTION SELECTION
# ============================================================

def deduplicate_connections(connections: List[ConnectionResult]) -> List[ConnectionResult]:
    seen: Set[Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[str, str], ...]]] = set()
    output: List[ConnectionResult] = []

    for conn in connections:
        key = (
            tuple(conn.property_ids),
            tuple(conn.qualifier_ids),
            tuple(conn.path_steps),
        )

        if key not in seen:
            seen.add(key)
            output.append(conn)

    return output


def choose_best_connections(connections: List[ConnectionResult]) -> List[ConnectionResult]:
    """
    Keep only the best-priority explanation type.

    Priority order:
      1. direct statement on Entity_1
      2. qualifier statement on Entity_1
      100. fallback graph path

    For Q503034 -> Q47221, this selects:
      Entity_1 statement qualifier:
        winner / for work

    and rejects fallback:
      winner / notable work
    """
    if not connections:
        return []

    best_priority = min(conn.priority for conn in connections)
    best = [conn for conn in connections if conn.priority == best_priority]

    best.sort(
        key=lambda c: (
            len(c.property_ids),
            len(c.qualifier_ids),
            c.property_ids,
            c.qualifier_ids,
            c.path_steps,
        )
    )

    return best


def find_best_connections(qid1: str, qid2: str) -> List[ConnectionResult]:
    """
    Main search function.

    Step 1:
      Search Entity_1's own Wikidata page.

    Step 2:
      Only if nothing is found on Entity_1's page, use fallback graph search.
    """
    entity1_results = find_connections_on_entity1_page(qid1, qid2)

    if entity1_results:
        return choose_best_connections(entity1_results)

    fallback_results = find_fallback_truthy_paths(
        qid1,
        qid2,
        max_hops=MAX_FALLBACK_HOPS,
    )

    return choose_best_connections(fallback_results)


def summarize_connections(
    connections: List[ConnectionResult],
    readable: bool,
    language: str = LABEL_LANGUAGE,
) -> Dict[str, Any]:
    if not connections:
        return {
            "Connection_Path": "",
            "Property_Number": "",
            "Property_list": "",
            "Qualifier_Number": "",
            "Qualifier_list": "",
        }

    path_strings: List[str] = []
    property_ids: List[str] = []
    qualifier_ids: List[str] = []

    for conn in connections:
        path_strings.append(
            build_path_string(
                conn.path_steps,
                readable=readable,
                language=language,
            )
        )
        property_ids.extend(conn.property_ids)
        qualifier_ids.extend(conn.qualifier_ids)

    property_ids = unique_preserve_order(property_ids)
    qualifier_ids = unique_preserve_order(qualifier_ids)

    if readable:
        property_list = ";".join(readable_property(pid, language) for pid in property_ids)
        qualifier_list = ";".join(readable_property(pid, language) for pid in qualifier_ids)
    else:
        property_list = ";".join(property_ids)
        qualifier_list = ";".join(qualifier_ids)

    return {
        "Connection_Path": "\n".join(unique_preserve_order(path_strings)),
        "Property_Number": len(property_ids) if property_ids else "",
        "Property_list": property_list,
        "Qualifier_Number": len(qualifier_ids) if qualifier_ids else "",
        "Qualifier_list": qualifier_list,
    }


# ============================================================
# CSV PROCESSING
# ============================================================

def read_input_rows(input_csv_path: str) -> List[Dict[str, str]]:
    with open(input_csv_path, "r", encoding="utf-8-sig", newline="") as infile:
        reader = csv.DictReader(infile)

        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")

        required = {"Entity_1", "Entity_2"}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

        return list(reader)


def process_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    processed_rows: List[Dict[str, str]] = []

    existence_cache: Dict[str, bool] = {}

    for row in tqdm(rows, desc="Processing rows", unit="row"):
        raw_entity_1 = row.get("Entity_1", "")
        raw_entity_2 = row.get("Entity_2", "")

        output_row = {
            "Entity_1": raw_entity_1,
            "Entity_2": raw_entity_2,
            "Connection_Path": "",
            "Property_Number": "",
            "Property_list": "",
            "Qualifier_Number": "",
            "Qualifier_list": "",
        }

        try:
            qid1 = validate_qid(extract_qid_from_cell(raw_entity_1))
            qid2 = validate_qid(extract_qid_from_cell(raw_entity_2))

            if qid1 not in existence_cache:
                existence_cache[qid1] = entity_exists(qid1)

            if qid2 not in existence_cache:
                existence_cache[qid2] = entity_exists(qid2)

            if not existence_cache[qid1]:
                raise ValueError(f"Entity_1 does not exist on Wikidata: {qid1}")

            if not existence_cache[qid2]:
                raise ValueError(f"Entity_2 does not exist on Wikidata: {qid2}")

            connections = find_best_connections(qid1, qid2)
            summary = summarize_connections(
                connections,
                readable=False,
                language=LABEL_LANGUAGE,
            )

            output_row.update(summary)

        except Exception as exc:
            print(
                f"Warning: could not process row "
                f"({raw_entity_1}, {raw_entity_2}): {exc}",
                file=sys.stderr,
            )

        processed_rows.append(output_row)

    return processed_rows


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


def write_output_label_csv(
    output_label_csv_path: str,
    rows: List[Dict[str, str]],
    language: str,
) -> None:
    fieldnames = [
        "Entity_1",
        "Entity_2",
        "Connection_Path",
        "Property_Number",
        "Property_list",
        "Qualifier_Number",
        "Qualifier_list",
    ]

    readable_rows: List[Dict[str, str]] = []

    for row in tqdm(rows, desc="Writing readable labels", unit="row"):
        readable_row = {
            "Entity_1": readable_cell_entity(row.get("Entity_1", ""), language),
            "Entity_2": readable_cell_entity(row.get("Entity_2", ""), language),
            "Connection_Path": readable_raw_connection_path(
                row.get("Connection_Path", ""),
                language,
            ),
            "Property_Number": row.get("Property_Number", ""),
            "Property_list": readable_pid_list(row.get("Property_list", ""), language),
            "Qualifier_Number": row.get("Qualifier_Number", ""),
            "Qualifier_list": readable_pid_list(row.get("Qualifier_list", ""), language),
        }

        readable_rows.append(readable_row)

    with open(output_label_csv_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(readable_rows)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    try:
        rows = read_input_rows(INPUT_CSV_PATH)
        processed_rows = process_rows(rows)

        write_output_csv(
            OUTPUT_CSV_PATH,
            processed_rows,
        )

        write_output_label_csv(
            OUTPUT_LABEL_CSV_PATH,
            processed_rows,
            language=LABEL_LANGUAGE,
        )

    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Done.")
    print(f"Raw output written to: {OUTPUT_CSV_PATH}")
    print(f"Readable-label output written to: {OUTPUT_LABEL_CSV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())