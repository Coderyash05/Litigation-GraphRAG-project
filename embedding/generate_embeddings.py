import os

from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase


# ============================================================
# 1. NEO4J CONFIGURATION
# ============================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"

NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

if not NEO4J_PASSWORD:
    raise ValueError(
        "NEO4J_PASSWORD is not set.\n"
        'Set it in PowerShell using:\n'
        '$env:NEO4J_PASSWORD="your_password"'
    )

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


# ============================================================
# 2. TEST NEO4J CONNECTION
# ============================================================

print("\nConnecting to Neo4j...")

try:
    driver.verify_connectivity()
    print("Neo4j connection successful.")

except Exception as e:
    driver.close()
    raise RuntimeError(
        f"Could not connect to Neo4j: {e}"
    )


# ============================================================
# 3. LOAD EMBEDDING MODEL
# ============================================================

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("\nLoading embedding model...")

model = SentenceTransformer(
    MODEL_NAME,
    device="cuda"
)

print("Embedding model loaded.")
print("Device:", model.device)
print("Embedding dimensions:", model.get_embedding_dimension())


# ============================================================
# 4. GET TAX ISSUES FROM KNOWLEDGE GRAPH
# ============================================================

print("\nRetrieving TaxIssue nodes from Neo4j...")

with driver.session() as session:

    result = session.run("""
        MATCH (t:TaxIssue)
        RETURN
            t.issue_id AS issue_id,
            t.name AS name,
            t.section AS section
        ORDER BY t.issue_id
    """)

    records = list(result)

print(f"Found {len(records)} TaxIssue nodes.")


# ============================================================
# 5. CHECK WHETHER TAX ISSUES EXIST
# ============================================================

if not records:

    print("\nNo TaxIssue nodes found in Neo4j.")
    print("Nothing to embed.")

    driver.close()
    raise SystemExit


# ============================================================
# 6. GENERATE AND STORE EMBEDDINGS
# ============================================================

print("\nGenerating embeddings...")

with driver.session() as session:

    for record in records:

        # Create a meaningful text representation
        # of the TaxIssue node.

        text = (
            f"Tax Issue: {record['name']}. "
            f"Section: {record['section']}."
        )

        print(f"\nProcessing {record['issue_id']}")
        print(f"Text: {text}")

        # Generate 384-dimensional embedding
        # using the GPU.

        embedding = model.encode(
            text,
            convert_to_numpy=True
        ).tolist()

        # Store the embedding inside the
        # corresponding Neo4j node.

        session.run("""
            MATCH (t:TaxIssue {issue_id: $issue_id})
            SET t.embedding = $embedding
        """,
        issue_id=record["issue_id"],
        embedding=embedding)

        print(
            f"Embedding stored for {record['issue_id']}"
        )


# ============================================================
# 7. VERIFY STORED EMBEDDINGS
# ============================================================

print("\nVerifying stored embeddings...")

with driver.session() as session:

    result = session.run("""
        MATCH (t:TaxIssue)
        RETURN
            t.issue_id AS issue_id,
            t.name AS name,
            CASE
                WHEN t.embedding IS NULL THEN 0
                ELSE size(t.embedding)
            END AS dimensions
        ORDER BY t.issue_id
    """)

    verification = list(result)


print("\nEmbedding Verification")
print("----------------------")

for record in verification:

    print(
        f"{record['issue_id']} | "
        f"{record['name']} | "
        f"Dimensions: {record['dimensions']}"
    )


# ============================================================
# 8. CLOSE NEO4J CONNECTION
# ============================================================

driver.close()

print("\nAll embeddings generated and stored successfully.")