import os
from neo4j import GraphDatabase
from huggingface_hub import InferenceClient


# ============================================================
# 1. NEO4J CONFIGURATION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError("NEO4J_PASSWORD is not set.")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


# ============================================================
# 2. HUGGING FACE CONFIGURATION
# ============================================================

HF_TOKEN = os.environ.get("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN is not set. "
        "Set it in PowerShell using: "
        '$env:HF_TOKEN="your_token"'
    )

llm = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)

MODEL = "deepseek-ai/DeepSeek-V3-0324"


# ============================================================
# 3. GET CASE INFORMATION FROM KNOWLEDGE GRAPH
# ============================================================

def get_case_info(case_id):

    with driver.session() as session:

        result = session.run("""
    MATCH (c:Case {case_id: $case_id})

    OPTIONAL MATCH (c)-[:CONCERNS]->(t:TaxIssue)

    OPTIONAL MATCH (c)-[:HEARD_BY]->(court:Court)

    OPTIONAL MATCH (c)-[:HAS_JUDGMENT]->(j:Judgment)

    OPTIONAL MATCH (j)-[:HAS_DOCUMENT]->(d:Document)

    OPTIONAL MATCH (company:Company)-[:INVOLVED_IN]->(c)

    OPTIONAL MATCH (t)-[:GOVERNED_BY]->(section:LegalSection)

    OPTIONAL MATCH (lawyer:Lawyer)-[:REPRESENTS]->(c)

    RETURN
        c.case_id AS case_id,
        c.title AS case_title,
        c.year AS year,
        company.name AS company,
        t.name AS tax_issue,
        t.section AS tax_section,
        section.section_number AS governing_section,
        court.name AS court,
        j.date AS judgment_date,
        j.outcome AS outcome,
        d.name AS document,
        lawyer.name AS lawyer
""", case_id=case_id)

        return result.single()


# ============================================================
# 4. ASK LLM TO UNDERSTAND THE USER'S QUESTION
# ============================================================

def extract_case_id(question):

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a simple information extraction assistant.

Your task is to identify the case ID from the user's question.

The case ID will look like:
CASE001
CASE002
CASE003

Return ONLY the case ID.

If no case ID is present, return:
NONE
"""
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=20,
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 5. ASK LLM TO FORMULATE THE FINAL ANSWER
# ============================================================

def generate_answer(question, record):

    context = f"""
Case ID: {record["case_id"]}
Case Title: {record["case_title"]}
Year: {record["year"]}
Company: {record["company"]}
Tax Issue: {record["tax_issue"]}
Tax Section: {record["tax_section"]}
Governing Section: {record["governing_section"]}
Court: {record["court"]}
Judgment Date: {record["judgment_date"]}
Outcome: {record["outcome"]}
Document: {record["document"]}
Lawyer: {record["lawyer"]}
"""

    response = llm.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are a legal-information assistant.

Answer the user's question using ONLY the
information provided in the Knowledge Graph context.

Do not invent facts.

If the information is not available,
say that it is not available in the Knowledge Graph.
"""
            },
            {
                "role": "user",
                "content": f"""
User question:
{question}

Knowledge Graph context:
{context}
"""
            }
        ],
        max_tokens=300,
        temperature=0
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 6. MAIN PROGRAM
# ============================================================

try:

    question = input("\nAsk a question: ")

    print("\nUnderstanding question...")

    case_id = extract_case_id(question)

    print("Detected Case ID:", case_id)

    if case_id == "NONE":

        print("\nI couldn't identify a Case ID.")

    else:

        print("\nSearching Knowledge Graph...")

        record = get_case_info(case_id)

        if not record:

            print(f"\nCase {case_id} was not found in the Knowledge Graph.")

        else:

            print("Information retrieved from Neo4j.")

            print("\nGenerating answer...")

            answer = generate_answer(question, record)

            print("\nANSWER")
            print("------------------------")
            print(answer)


finally:

    driver.close()