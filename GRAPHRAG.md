# GraphRAG — How It Works

This project answers questions about tax-litigation cases by combining two
kinds of knowledge, both stored in **one Neo4j database**:

- a **knowledge graph**, which holds structured facts: which company, which tax
  issue and legal section, which court or tribunal, which lawyer, and what the
  outcome was;
- **document chunks**, which hold the text of the judgment PDFs, split into
  chunks and embedded for semantic search through a Neo4j vector index.

For each question, the graph finds the relevant cases and supplies their facts.
A vector search then retrieves the judgment passages for those cases, and an
LLM writes an answer from both sources, citing where each statement came from.

---

## 1. Architecture

```
                        User question
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER  (graphrag_agent.py, LangGraph)         │
│                                                              │
│   identify_nodes ──► graph_context ──► document_context ──►  │
│        │                  │                  │               │
│        │                  │                  │    generate_answer
└────────┼──────────────────┼──────────────────┼────────┬──────┘
         │                  │                  │        │
         ▼                  ▼                  ▼        ▼
┌────────────────┐  ┌────────────────┐  ┌─────────────────┐  ┌──────────┐
│ GRAPH LAYER    │  │ GRAPH LAYER    │  │ EMBEDDING LAYER │  │ LLM      │
│ graph_layer.py │  │ graph_layer.py │  │ embedding/      │  │ DeepSeek │
│                │  │                │  │   store.py      │  │ V3 via   │
│ Which cases    │  │ Cypher query → │  │ Retrieval query │  │ Hugging  │
│ does the       │  │ case facts     │  │ → closest       │  │ Face     │
│ question name? │  │                │  │ judgment chunks │  │          │
└───────┬────────┘  └───────┬────────┘  └────────┬────────┘  └──────────┘
        └───────────────────┼────────────────────┘
                            ▼
              Neo4j (graph + chunks + vector index)
```

| Layer | File | Job |
|---|---|---|
| Orchestration | [graphrag_agent.py](graphrag_agent.py) | Runs the LangGraph workflow, calls the other layers, and builds the LLM prompts |
| Graph | [graph_layer.py](graph_layer.py) | Fixed, read-only Cypher queries against Neo4j |
| Embedding | [embedding/store.py](embedding/store.py) | PDF chunking, embeddings, ingestion into Neo4j and vector search |
| Embedding CLI | [embedding_layer.py](embedding_layer.py) | Command line for ingesting and searching documents |
| LLM | `ask_llm()` in [graphrag_agent.py](graphrag_agent.py) | `qwen/qwen3.8-27b` through the Groq API (`openai` client) |

### What reaches the LLM

The LLM never sees Cypher or embedding vectors. It sees only the **results**
those return, as text:

- **From the graph layer:** the Cypher query runs in Neo4j and returns facts
  such as `CASE001: ABC Ltd vs Income Tax Department (2024)` and
  `Tax issue: ...`. These are formatted as plain text.
- **From the embedding layer:** the retrieval query embeds the question, Neo4j
  returns the closest chunks, and their *text* is passed on together with the
  source (case, document, page).

Both go into a single prompt, and the LLM answers from that prompt alone.

---

## 2. The data

### Knowledge graph (Neo4j)

The graph must already exist in Neo4j. The graph layer only reads it; the
embedding layer adds `Document` and `DocumentChunk` nodes (see below). The
graph layer expects these nodes and relationships:

```
(Company)-[:INVOLVED_IN]->(Case)
(Lawyer)-[:REPRESENTS]->(Case)
(Case)-[:CONCERNS]->(TaxIssue)-[:GOVERNED_BY]->(LegalSection)
(Case)-[:HEARD_BY]->(Court)
(Case)-[:APPEALED_BEFORE]->(Tribunal)
(Case)-[:HAS_JUDGMENT]->(Judgment)-[:HAS_DOCUMENT]->(Document)
(Case)-[:RELATED_TO]-(Case)
```

Key properties: `Case.case_id`, `Case.title`, `Company.name`,
`TaxIssue.name`, `TaxIssue.section`, `LegalSection.section_number`,
`Lawyer.name`, `Court.name`, `Tribunal.name`, `Judgment.outcome`,
`Judgment.date`.

### Documents (Neo4j chunks)

The judgment PDFs are in [documents/](documents/), and
[documents/metadata.json](documents/metadata.json) links each PDF to its
`case_id` and `document_id`:

| PDF | case_id | document_id |
|---|---|---|
| CASE001_Judgment.pdf | CASE001 | DOC001 |
| CASE003_Judgment.pdf | CASE003 | DOC003 |

Ingestion stores each PDF as chunks next to the graph:

```
(DocumentChunk)-[:PART_OF]->(Document)-[:RELATES_TO]->(Case)
```

Each `DocumentChunk` also carries `case_id` and `document_id` as properties.
That is what connects the graph to the vector search: the graph returns case
IDs, and the search only looks at chunks with those IDs.

---

## 3. Ingestion (one-time setup)

Ingestion is done by `ingest_documents()` in
[embedding/store.py](embedding/store.py), which you run through
`python embedding_layer.py ingest`. For each PDF it:

1. **Reads each page** with `pypdf`, so every chunk keeps its page number.
2. **Splits the page into sections** at headings such as "Arguments of the
   Taxpayer" (`split_sections`). A heading is a short line that starts with a
   capital letter and doesn't end like a sentence.
3. **Packs whole sentences into chunks** of up to 1200 characters, with up to
   200 characters of overlap (`pack_sentences`). Sentences are never cut in half.
4. **Merges small neighbouring sections** that fit into one chunk, and merges a
   very short last chunk into the one before it (`chunk_page`).
5. **Embeds each chunk** with `BAAI/bge-small-en-v1.5` (384 dimensions). The
   embedded text starts with the case ID and title (`passage_text`), so a chunk
   that never names the company still matches questions that do.
6. **Writes the chunks to Neo4j.** It merges the `Document` node, links it to
   its `Case` (if the case is in the graph), deletes the document's old
   chunks, and creates one `DocumentChunk` per chunk with `case_id`,
   `document_id`, `page_number`, `section`, `text`, `embedding` and
   `model_name`. The vector index `chunk_embeddings` (cosine, 384 dimensions)
   is created on the first run.

Re-running ingestion is safe. A document is skipped when its
`ingest_signature` (a hash of the file, its metadata, the model and the chunk
settings) is unchanged. `--force` re-embeds anyway, and `--recreate` drops the
vector index and all embedded chunks first, which is needed when you switch to
a model with a different vector size.

> The old hand-written `DocumentChunk` nodes (e.g. `DOC001_P1_BG`) are deleted
> the first time their document is ingested and replaced by chunks from the
> real PDF.

---

## 4. Query time: the workflow step by step

`build_workflow()` in [graphrag_agent.py](graphrag_agent.py) compiles this
LangGraph:

```
START
  ↓
identify_nodes      Step 2.1  LLM + Neo4j: which nodes and cases?
  ↓
graph_context       Cypher query → case facts         (route: graph / both)
  ↓
document_context    Step 2.2  Vector retrieval query   (route: documents / both)
  ↓
generate_answer     Step 2.3  LLM answers from both, with citations
  ↓
END
```

Each step reads from and writes to a shared state dictionary
(`GraphRAGState`):

| Field | Written by | Contents |
|---|---|---|
| `question` | user | The question |
| `route` | identify_nodes | `graph`, `documents` or `both` |
| `nodes` | identify_nodes | Graph nodes the question refers to |
| `case_ids` | identify_nodes | Cases those nodes lead to |
| `case_reasons` | identify_nodes | Why each case was chosen |
| `graph_facts` | graph_context | Case facts as text |
| `chunks` | document_context | Retrieved judgment chunks |
| `answer` | generate_answer | Final answer |

### Step 2.1 — `identify_nodes`: find the cases

1. When the workflow is built, `node_catalogue()` loads every Case, Company,
   TaxIssue, LegalSection, Lawyer, Court and Tribunal from Neo4j into a
   numbered list. The graph is small, so the whole list fits in the prompt.
2. The LLM is shown that list and the question. It replies with JSON:
   - `nodes`: the numbers of the nodes the question refers to, for example a
     company name or a section;
   - `route`: where the answer should come from:
     - `graph` for structured facts only (who, which court, outcome, date);
     - `documents` for judgment content only (arguments, evidence, reasoning);
     - `both` when it needs both, or when the LLM is unsure (the default).
3. A safety net also adds any node whose name appears word for word in the
   question, in case the LLM missed it.
4. `cases_for_nodes()` runs one fixed Cypher query per node to find the
   connected cases. For example, a company leads to cases through
   `INVOLVED_IN`, and a section leads to cases through
   `CONCERNS → TaxIssue → GOVERNED_BY`.
5. **Fallback:** if no node matches, an unfiltered semantic search over all
   documents takes the cases of the 3 closest chunks instead.

### Graph context — `graph_context`: the Cypher query

`case_facts()` runs `CASE_FACTS_QUERY` for the chosen cases. It collects each
case's companies, tax issues and sections, courts, tribunals, lawyers,
judgments (outcome and date) and related cases. `format_case_facts()` then
turns the rows into text:

```
CASE001: ABC Ltd vs Income Tax Department (2024)
  Company: ABC Ltd (...)
  Tax issue: <issue> — <section> of the <act> (<description>)
  Court: High Court of Rajasthan
  Judgment: ...: <outcome> on <date> — document DOC001 (...)
```

This step is skipped when the route is `documents`.

### Step 2.2 — `document_context`: the retrieval query

For **each** chosen case, `semantic_search()`:

1. embeds the question with the same model used at ingestion, adding the BGE
   query prefix;
2. runs a Cypher query over only the `DocumentChunk` nodes with **that
   case's `case_id`**, scoring each with `vector.similarity.cosine`;
3. returns the closest chunks.

Because the graph has already narrowed the search to a few cases, this exact
comparison is cheap. The vector index (`db.index.vector.queryNodes`) is used
only for the unfiltered search in the Step 2.1 fallback. Scores are cosine
similarity scaled to 0–1, so 0.5 means unrelated.

Each case gets its own search (2 chunks per case, at most 6 in total), so when
a question compares two cases, one case can't take all the slots. The results
are sorted by similarity score, and each chunk is labelled with its source,
e.g. `CASE001, DOC001, page 2`.

This step is skipped when the route is `graph`.

### Step 2.3 — `generate_answer`: the LLM answer

One prompt combines:

- the question;
- the **knowledge graph facts** (or "Not retrieved for this question.");
- the **document excerpts**, each with its source and section (or "No document
  excerpts retrieved.").

The LLM is told to use only this context, not to invent facts, and to cite
every statement: `[CASE001, DOC001, page 2]` for document excerpts and
`[Knowledge Graph]` for graph facts. `temperature=0` keeps answers repeatable.

### Routing

```
identify_nodes ──route = documents──────────────────► document_context
      │                                                      ▲
      └──route = graph / both──► graph_context ──both────────┘
                                      │
                                      └──graph──► generate_answer
```

---

## 5. Running it

### Prerequisites

- Python environment with the project dependencies (`.venv312`)
- Neo4j 5.18 or later running at `bolt://localhost:7687`, with the graph
  loaded. This is the only database.
- A Groq API key (free tier at console.groq.com) for the LLM

### Setup

Put your private settings in the `.env` file in the project root (it is in
`.gitignore`, so it is never committed):

```
NEO4J_PASSWORD=your_neo4j_password
GROQ_API_KEY=your_groq_key
# Optional: NEO4J_URI and NEO4J_USERNAME default to bolt://localhost:7687 and neo4j
```

`graph_layer.py` loads this file on import, so both scripts pick it up.
Variables already set in the shell take precedence.

```powershell
# One-time: embed the PDFs in documents/ into Neo4j
python embedding_layer.py ingest
```

### Ask a question

```powershell
# Single question
python graphrag_agent.py "What evidence did ABC Ltd provide?"

# Interactive mode: keeps asking until you enter a blank line
python graphrag_agent.py
```

The output shows every step: the identified nodes and route, the graph facts,
the retrieved chunks with their scores, and the answer.

---

## 6. Sample prompts

The walk-throughs below show what each step does. The exact scores and wording
depend on the data and the LLM.

### Example 1 — a question about judgment content

```powershell
python graphrag_agent.py "What evidence did ABC Ltd provide to support its business expenses?"
```

1. **identify_nodes:** the LLM picks `Company: ABC Ltd`. The route is `both`
   (or `documents`), because evidence is in the judgment text.
   `cases_for_nodes` follows `INVOLVED_IN` to **CASE001**.
2. **graph_context:** facts for CASE001: the company, the tax issue and
   section, the court, and the judgment outcome.
3. **document_context:** only CASE001's chunks are searched, returning
   the chunks about the evidence produced, expected on page 2 of DOC001.
4. **generate_answer:** an answer that lists the evidence, with each point
   cited as `[CASE001, DOC001, page 2]` and the case facts as
   `[Knowledge Graph]`.

The output looks like this:

```
========================================
STEP 2.1 — IDENTIFIED NODES
========================================
Route: both
Nodes: Company: ABC Ltd
Case CASE001 <- Company: ABC Ltd

========================================
GRAPH CONTEXT (Neo4j)
========================================
CASE001: ABC Ltd vs Income Tax Department (2024)
  ...

========================================
STEP 2.2 — DOCUMENT CONTEXT (vector search)
========================================
0.8xx  [CASE001, DOC001, page 2] ...
...

========================================
STEP 2.3 — ANSWER
========================================
ABC Ltd produced ... [CASE001, DOC001, page 2] ...
```

### Example 2 — a question spanning two cases

```powershell
python graphrag_agent.py "Compare the outcomes of the ABC Ltd and DEF Industries cases and which court decided each"
```

1. **identify_nodes:** the LLM picks both companies, which lead to **CASE001**
   and **CASE003**. The route is likely `graph`, since outcomes and courts are
   graph facts.
2. **graph_context:** facts for both cases, including the High Court of
   Rajasthan for CASE001, the Income Tax Appellate Tribunal (Mumbai Bench) for
   CASE003, and each judgment's outcome and date.
3. **document_context:** skipped if the route is `graph`. If the route is
   `both`, the chunks are searched once per case, so each case gets its own.
4. **generate_answer:** a side-by-side comparison cited to
   `[Knowledge Graph]`.

### More questions to try

- `Which cases concern Section 32?` — found through the LegalSection node
- `What did the Revenue say about trial runs of the machines?` — no node is
  named, so the semantic fallback finds CASE003
- `Why did the Assessing Officer reduce DEF Industries' depreciation claim?`

---

## 7. Configuration

| Setting | Where | Default |
|---|---|---|
| `LLM_MODEL` | graphrag_agent.py | `deepseek-ai/DeepSeek-V3-0324` |
| `EMBEDDING_CONFIG` | graphrag_agent.py | `EmbeddingConfig()`: bge-small, chunk size 1200, overlap 200 |
| `CHUNKS_PER_CASE` / `MAX_CHUNKS` | graphrag_agent.py | 2 / 6 |
| `DISCOVERY_CHUNKS` | graphrag_agent.py | 3 |
| `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD` | `.env` | `bolt://localhost:7687`, `neo4j`, required |
| `GROQ_API_KEY` | `.env` | required |
| `EMBEDDING_DEVICE` | environment | `cpu` (`cuda` to use the GPU) |
| `VECTOR_INDEX` | embedding/store.py | `chunk_embeddings` |

`EMBEDDING_CONFIG` must match the settings the documents were ingested with.
The search only looks at chunks whose `model_name` matches, so with a
different model it would find nothing.

---

## 8. Known limitations

- **Routing can drop a source.** If the LLM chooses `graph` or `documents`,
  the other source never reaches the answer step.
- **Citations aren't checked** against the chunks that were actually retrieved.
- **The semantic fallback has no score threshold**, so a question unrelated to
  any case still gets the closest cases.
- **Outcome and aggregate questions** (e.g. "Which cases did the taxpayer
  win?") can't be targeted, because Judgment nodes aren't in the node
  catalogue.
- **No conversation memory:** each question in interactive mode stands alone.
- **No end-to-end evaluation yet.** [evaluation/queries.json](evaluation/queries.json)
  measures retrieval only, not node identification, routing or answer quality.
