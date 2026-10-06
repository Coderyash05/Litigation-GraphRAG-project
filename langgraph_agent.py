import os
import re
from typing import TypedDict

from neo4j import GraphDatabase
from huggingface_hub import InferenceClient

from langgraph.graph import StateGraph, START, END


# ============================================================
# 1. CONFIGURATION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"

NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
HF_TOKEN = os.environ.get("HF_TOKEN")

MODEL = "deepseek-ai/DeepSeek-V3-0324"


if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD is not set."
    )

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set."
    )


# ============================================================
# 2. CONNECTIONS
# ============================================================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)

llm = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


# ============================================================
# 3. LANGGRAPH STATE
# ============================================================

class AgentState(TypedDict):
    question: str
    cypher_query: str
    graph_results: str
    final_answer: str


# ============================================================
# 4. NEO4J SCHEMA
# ============================================================

SCHEMA = """
NODE LABELS AND PROPERTIES:

Case:
- case_id
- title
- year

Company:
- company_id
- name
- industry
- location

TaxIssue:
- issue_id
- name
- section

LegalSection:
- section_id
- section_number
- act
- description

Court:
- name
- location

Lawyer:
- lawyer_id
- name
- role

Tribunal:
- name
- location

Judgment:
- judgment_id
- date
- outcome

Document:
- document_id
- name
- document_type
- source


RELATIONSHIPS:

(Company)-[:INVOLVED_IN]->(Case)

(Case)-[:CONCERNS]->(TaxIssue)

(TaxIssue)-[:GOVERNED_BY]->(LegalSection)

(Case)-[:HEARD_BY]->(Court)

(Lawyer)-[:REPRESENTS]->(Case)

(Case)-[:APPEALED_BEFORE]->(Tribunal)

(Case)-[:HAS_JUDGMENT]->(Judgment)

(Judgment)-[:HAS_DOCUMENT]->(Document)

(Case)-[:RELATED_TO]->(Case)


IMPORTANT PROPERTY RULES:

Case uses:
    case_id

TaxIssue uses:
    name

LegalSection uses:
    section_number

Company uses:
    name

Lawyer uses:
    name

Court uses:
    name

Tribunal uses:
    name

Judgment uses:
    date
    outcome

Document uses:
    name
    document_type
    source

Do NOT invent properties.
"""


# ============================================================
# 5. NODE: GENERATE CYPHER
# ============================================================

def generate_cypher(state: AgentState):

    question = state["question"]

    prompt = f"""
You are a Neo4j Cypher query generator.

Your task is to convert a natural-language question
into a READ-ONLY Cypher query.

Here is the exact Neo4j schema:

{SCHEMA}

STRICT RULES:

1. Use ONLY the labels, properties and relationships
   defined in the schema.

2. Never invent property names.

3. Case IDs MUST use:
   case_id

4. TaxIssue name MUST use:
   name

5. LegalSection number MUST use:
   section_number

6. Company name MUST use:
   name

7. Lawyer name MUST use:
   name

8. Use OPTIONAL MATCH when information may not exist.

9. Only use read operations:
   MATCH
   OPTIONAL MATCH
   WHERE
   RETURN
   WITH
   ORDER BY
   LIMIT

10. NEVER use:
   CREATE
   MERGE
   DELETE
   DETACH DELETE
   SET
   REMOVE
   DROP
   LOAD CSV

11. Return ONLY the Cypher query.

12. Do not use markdown.

User question:
{question}
"""

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0
    )

    cypher = response.choices[0].message.content.strip()

    # Remove accidental markdown fences
    cypher = cypher.replace("```cypher", "")
    cypher = cypher.replace("```", "")
    cypher = cypher.strip()

    return {
        "cypher_query": cypher
    }


# ============================================================
# 6. CYPHER SAFETY VALIDATION
# ============================================================

def validate_cypher(cypher: str):

    forbidden_keywords = [
        "CREATE",
        "MERGE",
        "DELETE",
        "DETACH",
        "SET",
        "REMOVE",
        "DROP",
        "LOAD CSV",
        "ALTER",
        "GRANT",
        "DENY",
        "REVOKE"
    ]

    upper_query = cypher.upper()

    for keyword in forbidden_keywords:

        if re.search(r"\b" + re.escape(keyword) + r"\b", upper_query):
            return False

    # Require a read query
    if "MATCH" not in upper_query:
        return False

    if "RETURN" not in upper_query:
        return False

    # Prevent multiple statements
    if ";" in cypher.rstrip(";"):
        return False

    return True


# ============================================================
# 7. NODE: EXECUTE CYPHER
# ============================================================

def execute_cypher(state: AgentState):

    cypher = state["cypher_query"]

    print("\nExecuting Cypher...")

    print("-------------------------")
    print(cypher)

    # Validate query
    if not validate_cypher(cypher):

        return {
            "graph_results":
                "The generated Cypher query failed safety validation."
        }

    try:

        with driver.session() as session:

            result = session.run(cypher)

            records = list(result)

            if not records:

                return {
                    "graph_results":
                        "No information found in the Knowledge Graph."
                }

            results = []

            for record in records:

                results.append(record.data())

            return {
                "graph_results": str(results)
            }

    except Exception as e:

        return {
            "graph_results":
                f"Neo4j query error: {str(e)}"
        }


# ============================================================
# 8. NODE: GENERATE FINAL ANSWER
# ============================================================

def generate_answer(state: AgentState):

    question = state["question"]

    graph_results = state["graph_results"]

    prompt = f"""
You are a legal-information assistant.

Answer the user's question using ONLY the
information retrieved from the Knowledge Graph.

Do not invent facts.

If a particular piece of information is missing,
clearly say that it is not available.

User question:
{question}

Knowledge Graph results:
{graph_results}

Give a concise and factual answer.
"""

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=500,
        temperature=0
    )

    answer = response.choices[0].message.content.strip()

    return {
        "final_answer": answer
    }


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

workflow = StateGraph(AgentState)


workflow.add_node(
    "generate_cypher",
    generate_cypher
)

workflow.add_node(
    "execute_cypher",
    execute_cypher
)

workflow.add_node(
    "generate_answer",
    generate_answer
)


# Workflow edges

workflow.add_edge(
    START,
    "generate_cypher"
)

workflow.add_edge(
    "generate_cypher",
    "execute_cypher"
)

workflow.add_edge(
    "execute_cypher",
    "generate_answer"
)

workflow.add_edge(
    "generate_answer",
    END
)


# Compile graph

app = workflow.compile()


# ============================================================
# 10. RUN THE AGENT
# ============================================================

try:

    question = input("\nAsk a question: ")

    result = app.invoke(
        {
            "question": question,
            "cypher_query": "",
            "graph_results": "",
            "final_answer": ""
        }
    )

    print("\n")
    print("========================================")
    print("GENERATED CYPHER")
    print("========================================")

    print(result["cypher_query"])


    print("\n")
    print("========================================")
    print("GRAPH RESULTS")
    print("========================================")

    print(result["graph_results"])


    print("\n")
    print("========================================")
    print("FINAL ANSWER")
    print("========================================")

    print(result["final_answer"])


finally:

    driver.close()