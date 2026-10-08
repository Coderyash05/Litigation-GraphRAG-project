"""
GraphRAG orchestrator (LangGraph).

The graph layer and the embedding layer both read from one Neo4j
database and answer to this orchestration layer, which asks the
LLM for a grounded answer:

    START
      ↓
    identify_nodes     Step 2.1  LLM picks graph nodes named in the
      ↓                          question; Neo4j maps them to cases
    graph_context      Neo4j facts for those cases        (route: graph / both)
      ↓
    document_context   Step 2.2  Vector search, filtered   (route: documents / both)
      ↓                          to those cases
    generate_answer    Step 2.3  LLM answers from both, citing sources
      ↓
    END

Run from the project root (NEO4J_PASSWORD and GROQ_API_KEY are read
from the .env file):

    python graphrag_agent.py
    python graphrag_agent.py "What evidence did ABC Ltd provide?"
"""

import json
import os
import re
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from openai import OpenAI

from embedding.store import EmbeddingConfig, semantic_search
from graph_layer import (
    case_facts,
    cases_for_nodes,
    format_case_facts,
    get_driver,
    node_catalogue
)


# ============================================================
# CONFIGURATION
# ============================================================

# Groq's API (OpenAI-compatible, free tier)
LLM_BASE_URL = "https://api.groq.com/openai/v1"
LLM_MODEL = "qwen/qwen3.8-27b"

# Same settings the documents were ingested with
EMBEDDING_CONFIG = EmbeddingConfig()

# Chunks to retrieve per identified case (max 6 in total)
CHUNKS_PER_CASE = 2
MAX_CHUNKS = 6

# When no graph node matches, cases are taken from an
# unfiltered semantic search instead
DISCOVERY_CHUNKS = 3

ROUTES = ("graph", "documents", "both")


# ============================================================
# STATE
# ============================================================

class GraphRAGState(TypedDict, total=False):
    question: str
    route: str
    nodes: list          # identified graph nodes
    case_ids: list       # cases the nodes lead to
    case_reasons: dict   # case_id -> how it was identified
    graph_facts: str
    chunks: list         # retrieved document chunks
    answer: str


# ============================================================
# LLM
# ============================================================

def ask_llm(llm, prompt, max_tokens=600):

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0
    )

    return response.choices[0].message.content.strip()


def parse_json(text):
    """
    First {...} object in the LLM reply, or {} if none parses.
    """

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        return {}

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


# ============================================================
# WORKFLOW
# ============================================================

def build_workflow(driver, llm):

    catalogue = node_catalogue(driver)

    # ------------------------------------------------------------
    # STEP 2.1 — IDENTIFY NODES
    # ------------------------------------------------------------

    def identify_nodes(state: GraphRAGState):

        question = state["question"]

        catalogue_lines = "\n".join(
            f"[{index}] {node['label']}: {node['key']}"
            + (f" ({node['description']})" if node["description"] else "")
            for index, node in enumerate(catalogue)
        )

        reply = ask_llm(
            llm,
            f"""
You connect a user's question to nodes of a litigation knowledge graph.

Graph nodes:
{catalogue_lines}

Tasks:
1. List the numbers of the nodes the question refers to, directly
   or indirectly (e.g. a company name inside a case title means
   that case). Only use numbers from the list. Use [] if none.
2. Choose where the answer will come from:
   "graph"     - structured facts only (which company, section,
                 court, lawyer, outcome, date, related cases)
   "documents" - contents of judgments only (arguments, evidence,
                 reasoning, findings)
   "both"      - needs both, or unsure

Question: {question}

Reply with JSON only, e.g. {{"nodes": [0, 2], "route": "both"}}
""",
            max_tokens=100
        )

        parsed = parse_json(reply)

        nodes = [
            catalogue[index]
            for index in parsed.get("nodes", [])
            if isinstance(index, int) and 0 <= index < len(catalogue)
        ]

        # Also keep nodes whose name appears word-for-word in the
        # question, in case the LLM missed one
        lowered = question.lower()

        for node in catalogue:
            if node["key"].lower() in lowered and node not in nodes:
                nodes.append(node)

        route = parsed.get("route")

        if route not in ROUTES:
            route = "both"

        case_reasons = cases_for_nodes(driver, nodes)

        # Nothing in the graph matched: find the cases whose
        # documents are semantically closest instead
        if not case_reasons:

            for result in semantic_search(
                driver,
                EMBEDDING_CONFIG,
                question,
                limit=DISCOVERY_CHUNKS
            ):
                case_reasons.setdefault(
                    result["case_id"],
                    [f"semantic match ({result['score']:.2f})"]
                )

        return {
            "route": route,
            "nodes": [f"{node['label']}: {node['key']}" for node in nodes],
            "case_ids": sorted(case_reasons),
            "case_reasons": case_reasons
        }

    # ------------------------------------------------------------
    # GRAPH CONTEXT
    # ------------------------------------------------------------

    def graph_context(state: GraphRAGState):

        return {
            "graph_facts": format_case_facts(
                case_facts(driver, state["case_ids"])
            )
        }

    # ------------------------------------------------------------
    # STEP 2.2 — DOCUMENT CONTEXT (targeted semantic search)
    # ------------------------------------------------------------

    def document_context(state: GraphRAGState):

        case_ids = state["case_ids"]

        if not case_ids:
            return {"chunks": []}

        # Search each case separately, so that when the question
        # involves several cases one of them cannot take all the
        # slots (e.g. comparing two cases' outcomes).
        per_case = max(1, min(CHUNKS_PER_CASE, MAX_CHUNKS // len(case_ids)))

        results = []

        for case_id in case_ids:
            results.extend(
                semantic_search(
                    driver,
                    EMBEDDING_CONFIG,
                    state["question"],
                    limit=per_case,
                    case_id=case_id
                )
            )

        results.sort(key=lambda result: result["score"], reverse=True)

        return {
            "chunks": [
                {
                    "source": (
                        f"{result['case_id']}, "
                        f"{result['document_id']}, "
                        f"page {result['page_number']}"
                    ),
                    "section": result["section"],
                    "score": result["score"],
                    "text": result["text"]
                }
                for result in results
            ]
        }

    # ------------------------------------------------------------
    # STEP 2.3 — COMBINED ANSWER
    # ------------------------------------------------------------

    def generate_answer(state: GraphRAGState):

        chunks = state.get("chunks", [])

        excerpts = "\n\n".join(
            f"[{chunk['source']}] ({chunk['section']})\n{chunk['text']}"
            for chunk in chunks
        ) or "No document excerpts retrieved."

        answer = ask_llm(
            llm,
            f"""
You are a legal-information assistant for tax litigation.

Answer the question using ONLY the knowledge graph facts and the
document excerpts below. Do not use outside knowledge and do not
invent facts. If something is not in the context, say it is not
available.

After each statement taken from a document excerpt, cite its source
in square brackets exactly as given, e.g. [CASE001, DOC001, page 2].
Cite graph facts as [Knowledge Graph].

Question:
{state["question"]}

Knowledge graph facts:
{state.get("graph_facts") or "Not retrieved for this question."}

Document excerpts:
{excerpts}

Give a concise, factual answer.
""",
            max_tokens=600
        )

        return {"answer": answer}

    # ------------------------------------------------------------
    # ROUTING
    # ------------------------------------------------------------

    def after_identify(state: GraphRAGState):

        if state["route"] == "documents":
            return "document_context"

        return "graph_context"

    def after_graph(state: GraphRAGState):

        if state["route"] == "graph":
            return "generate_answer"

        return "document_context"

    workflow = StateGraph(GraphRAGState)

    workflow.add_node("identify_nodes", identify_nodes)
    workflow.add_node("graph_context", graph_context)
    workflow.add_node("document_context", document_context)
    workflow.add_node("generate_answer", generate_answer)

    workflow.add_edge(START, "identify_nodes")

    workflow.add_conditional_edges(
        "identify_nodes",
        after_identify,
        ["graph_context", "document_context"]
    )

    workflow.add_conditional_edges(
        "graph_context",
        after_graph,
        ["document_context", "generate_answer"]
    )

    workflow.add_edge("document_context", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ============================================================
# OUTPUT
# ============================================================

def print_result(result):

    print("\n========================================")
    print("STEP 2.1 — IDENTIFIED NODES")
    print("========================================")
    print("Route:", result["route"])
    print("Nodes:", ", ".join(result["nodes"]) or "none")

    for case_id, reasons in result["case_reasons"].items():
        print(f"Case {case_id} <- {', '.join(reasons)}")

    if result.get("graph_facts"):
        print("\n========================================")
        print("GRAPH CONTEXT (Neo4j)")
        print("========================================")
        print(result["graph_facts"])

    if result.get("chunks") is not None and result["route"] != "graph":
        print("\n========================================")
        print("STEP 2.2 — DOCUMENT CONTEXT (vector search)")
        print("========================================")

        for chunk in result["chunks"]:
            print(
                f"{chunk['score']:.3f}  [{chunk['source']}] "
                f"{chunk['section']}"
            )

    print("\n========================================")
    print("STEP 2.3 — ANSWER")
    print("========================================")
    print(result["answer"])


# ============================================================
# MAIN
# ============================================================

def create_llm():

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set.\n"
            "Add GROQ_API_KEY=your_key to the .env file."
        )

    return OpenAI(api_key=api_key, base_url=LLM_BASE_URL)


def main():

    llm = create_llm()

    driver = get_driver()

    try:

        app = build_workflow(driver, llm)

        # Question on the command line: answer it and exit
        question = " ".join(sys.argv[1:]).strip()

        if question:
            print_result(app.invoke({"question": question}))
            return

        while True:

            question = input(
                "\nAsk a question (blank to exit): "
            ).strip()

            if not question:
                break

            print_result(app.invoke({"question": question}))

    finally:
        driver.close()


if __name__ == "__main__":
    main()
