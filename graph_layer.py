"""
Graph layer: structured case knowledge from Neo4j.

Used by the GraphRAG orchestrator for:

    Step 2.1  node_catalogue() + cases_for_nodes()
              which nodes does the question refer to, and which
              cases are they connected to?

    Graph context  case_facts()
              company, tax issue, legal section, court, lawyer,
              judgment and related cases for those cases.

All queries are fixed and read-only; no LLM-generated Cypher.
"""

import os
from pathlib import Path

from neo4j import READ_ACCESS, GraphDatabase


# ============================================================
# CONFIGURATION
# ============================================================

# Private settings (NEO4J_PASSWORD, GROQ_API_KEY, ...) as KEY=value
# lines. Kept out of git by .gitignore.
ENV_FILE = Path(__file__).parent / ".env"


def load_env_file(path=ENV_FILE):
    """
    Copy KEY=value lines from the .env file into os.environ.
    Variables already set in the shell take precedence.
    """

    if not path.exists():
        return

    for line in path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)

        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


# Runs on import, so every script that uses the graph layer
# (graphrag_agent.py, embedding_layer.py) gets the settings
load_env_file()

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")


def get_driver():

    password = os.environ.get("NEO4J_PASSWORD")

    if not password:
        raise ValueError(
            "NEO4J_PASSWORD is not set.\n"
            "Add NEO4J_PASSWORD=your_password to the .env file."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USERNAME, password)
    )

    driver.verify_connectivity()

    return driver


def read(driver, cypher, **parameters):
    """
    Run a query in a read-only session.
    """

    with driver.session(default_access_mode=READ_ACCESS) as session:
        return [record.data() for record in session.run(cypher, parameters)]


# ============================================================
# STEP 2.1 — NODE IDENTIFICATION
# ============================================================

# Every node a question can refer to, as (label, key).
# The graph is small, so the whole catalogue can be shown to
# the LLM, which picks the nodes the question refers to.
CATALOGUE_QUERY = """
MATCH (c:Case)
RETURN 'Case' AS label, c.case_id AS key, c.title AS description
UNION
MATCH (n:Company)
RETURN 'Company' AS label, n.name AS key, n.industry AS description
UNION
MATCH (n:TaxIssue)
RETURN 'TaxIssue' AS label, n.name AS key, n.section AS description
UNION
MATCH (n:LegalSection)
RETURN 'LegalSection' AS label, n.section_number AS key,
       n.description AS description
UNION
MATCH (t:TaxIssue) WHERE t.section IS NOT NULL
RETURN 'LegalSection' AS label, t.section AS key,
       null AS description
UNION
MATCH (n:Lawyer)
RETURN 'Lawyer' AS label, n.name AS key, n.role AS description
UNION
MATCH (n:Court)
RETURN 'Court' AS label, n.name AS key, n.location AS description
UNION
MATCH (n:Tribunal)
RETURN 'Tribunal' AS label, n.name AS key, n.location AS description
"""


# For each label: the cases connected to the node with that key
CASES_FOR_NODE = {
    "Case": """
        MATCH (c:Case {case_id: $key})
        RETURN c.case_id AS case_id
    """,
    "Company": """
        MATCH (:Company {name: $key})-[:INVOLVED_IN]->(c:Case)
        RETURN c.case_id AS case_id
    """,
    "TaxIssue": """
        MATCH (c:Case)-[:CONCERNS]->(:TaxIssue {name: $key})
        RETURN c.case_id AS case_id
    """,
    # A section can be a LegalSection node or only a TaxIssue
    # property (e.g. Section 32 for CASE003)
    "LegalSection": """
        MATCH (c:Case)-[:CONCERNS]->(t:TaxIssue)
        OPTIONAL MATCH (t)-[:GOVERNED_BY]->(s:LegalSection)
        WITH c, t, s
        WHERE t.section = $key OR s.section_number = $key
        RETURN DISTINCT c.case_id AS case_id
    """,
    "Lawyer": """
        MATCH (:Lawyer {name: $key})-[:REPRESENTS]->(c:Case)
        RETURN c.case_id AS case_id
    """,
    "Court": """
        MATCH (c:Case)-[:HEARD_BY]->(:Court {name: $key})
        RETURN c.case_id AS case_id
    """,
    "Tribunal": """
        MATCH (c:Case)-[:APPEALED_BEFORE]->(:Tribunal {name: $key})
        RETURN c.case_id AS case_id
    """
}


def node_catalogue(driver):
    """
    [{"label": "Company", "key": "ABC Ltd", "description": ...}, ...]
    """

    rows = read(driver, CATALOGUE_QUERY)

    unique = {}

    for row in rows:

        if row["key"]:
            unique.setdefault((row["label"], row["key"]), row)

    return list(unique.values())


def cases_for_nodes(driver, nodes):
    """
    nodes: [{"label": ..., "key": ...}]

    Returns {case_id: ["Company: ABC Ltd", ...]} — each case with
    the identified nodes that led to it.
    """

    cases = {}

    for node in nodes:

        cypher = CASES_FOR_NODE.get(node["label"])

        if not cypher:
            continue

        for row in read(driver, cypher, key=node["key"]):
            cases.setdefault(row["case_id"], []).append(
                f"{node['label']}: {node['key']}"
            )

    return cases


# ============================================================
# GRAPH CONTEXT
# ============================================================

CASE_FACTS_QUERY = """
MATCH (c:Case) WHERE c.case_id IN $case_ids
RETURN
    c.case_id AS case_id,
    c.title AS title,
    c.year AS year,
    COLLECT {
        MATCH (co:Company)-[:INVOLVED_IN]->(c)
        RETURN co.name + coalesce(' (' + co.industry + ', ' + co.location + ')', '')
    } AS companies,
    COLLECT {
        MATCH (c)-[:CONCERNS]->(t:TaxIssue)
        OPTIONAL MATCH (t)-[:GOVERNED_BY]->(s:LegalSection)
        RETURN t.name
            + coalesce(' — ' + s.section_number + ' of the ' + s.act
                       + ' (' + s.description + ')',
                       ' — ' + t.section, '')
    } AS tax_issues,
    COLLECT {
        MATCH (c)-[:HEARD_BY]->(court:Court)
        RETURN court.name
    } AS courts,
    COLLECT {
        MATCH (c)-[:APPEALED_BEFORE]->(tr:Tribunal)
        RETURN tr.name + coalesce(', ' + tr.location, '')
    } AS tribunals,
    COLLECT {
        MATCH (l:Lawyer)-[:REPRESENTS]->(c)
        RETURN l.name + coalesce(' (' + l.role + ')', '')
    } AS lawyers,
    COLLECT {
        MATCH (c)-[:HAS_JUDGMENT]->(j:Judgment)
        OPTIONAL MATCH (j)-[:HAS_DOCUMENT]->(d:Document)
        RETURN j.judgment_id + ': ' + coalesce(j.outcome, 'outcome unknown')
            + ' on ' + coalesce(j.date, 'unknown date')
            + coalesce(' — document ' + d.document_id + ' (' + d.name + ')', '')
    } AS judgments,
    COLLECT {
        MATCH (c)-[:RELATED_TO]-(r:Case)
        RETURN r.case_id + ': ' + r.title
    } AS related_cases
ORDER BY case_id
"""


def case_facts(driver, case_ids):
    """
    Structured facts for each case, one dict per case.
    """

    if not case_ids:
        return []

    return read(driver, CASE_FACTS_QUERY, case_ids=list(case_ids))


def format_case_facts(facts):
    """
    Plain-text version of case_facts() for the LLM prompt.
    """

    blocks = []

    for case in facts:

        lines = [f"{case['case_id']}: {case['title']} ({case['year']})"]

        for field, label in [
            ("companies", "Company"),
            ("tax_issues", "Tax issue"),
            ("courts", "Court"),
            ("tribunals", "Tribunal"),
            ("lawyers", "Lawyer"),
            ("judgments", "Judgment"),
            ("related_cases", "Related case")
        ]:
            for value in case[field]:
                lines.append(f"  {label}: {value}")

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) or "No graph facts found."