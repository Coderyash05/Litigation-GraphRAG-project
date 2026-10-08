"""
Litigation AI Assistant: presentation front end.

Runs the same GraphRAG workflow as graphrag_agent.py, but shows
each question's progress live and lays the result out for a
non-technical audience: the answer first, then how it was found
(which cases, which case-record facts, which judgment passages).

Run from the project root (Neo4j must be running):

    python main.py
    python main.py "Who represented ABC Ltd?"

In interactive mode you can also type:

    help      sample questions (type a number to ask one)
    about     how the system works, in plain language
    exit      quit (a blank line also quits)
"""

import re
import sys

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from huggingface_hub.utils import disable_progress_bars
from huggingface_hub.utils import logging as hf_logging
from transformers.utils import logging as transformers_logging

from embedding.store import get_model
from graph_layer import case_facts, get_driver
from graphrag_agent import (
    EMBEDDING_CONFIG,
    LLM_MODEL,
    build_workflow,
    create_llm
)

# Keep library warnings and progress bars out of the presentation
# (e.g. a stale saved Hugging Face login, "Loading weights" bars)
hf_logging.set_verbosity_error()
disable_progress_bars()
transformers_logging.set_verbosity_error()
transformers_logging.disable_progress_bar()


# ============================================================
# CONFIGURATION
# ============================================================

SAMPLE_QUESTIONS = [
    "What was the litigation case faced by ABC Ltd? Who was the "
    "lawyer that represented the company and what was the "
    "court's judgement?",
    "Who represented ABC Ltd?",
    "What evidence did ABC Ltd provide?",
    "Which cases concern Section 32?",
    "Why did the Assessing Officer reduce DEF Industries' "
    "depreciation claim?",
    "What did the Revenue say about trial runs of the machines?"
]

# Plain-language explanation of each route
ROUTE_TEXT = {
    "graph": (
        "Case records only",
        "The question asks for facts such as parties, lawyers, "
        "courts, dates or outcomes, so the structured case "
        "records are enough."
    ),
    "documents": (
        "Judgment text only",
        "The question asks what a judgment says (arguments, "
        "evidence, reasoning), so the judgment documents were "
        "searched."
    ),
    "both": (
        "Case records + judgment text",
        "The question needs structured facts and what the "
        "judgment says, so both sources were used."
    )
}

# Progress lines, in workflow order: (node, done label, running label)
STEPS = [
    ("identify_nodes", "Understood the question",
     "Understanding the question"),
    ("graph_context", "Looked up case records",
     "Looking up case records"),
    ("document_context", "Searched judgment documents",
     "Searching judgment documents"),
    ("generate_answer", "Wrote the answer",
     "Writing the answer")
]

# Which steps run on each route
STEPS_FOR_ROUTE = {
    "graph": {"identify_nodes", "graph_context", "generate_answer"},
    "documents": {"identify_nodes", "document_context", "generate_answer"},
    "both": {"identify_nodes", "graph_context", "document_context",
             "generate_answer"}
}

# Field labels for the case-record table
FACT_FIELDS = [
    ("companies", "Company"),
    ("tax_issues", "Tax issue"),
    ("courts", "Court"),
    ("tribunals", "Tribunal"),
    ("lawyers", "Lawyer"),
    ("judgments", "Judgment"),
    ("related_cases", "Related case")
]

console = Console()


# ============================================================
# RUNNING A QUESTION
# ============================================================

def step_detail(node, state):
    """
    Short result shown next to a finished progress step.
    """

    if node == "identify_nodes":
        cases = state.get("case_ids", [])
        return (
            f"{ROUTE_TEXT[state['route']][0]} · "
            f"{len(cases)} case(s) identified"
        )

    if node == "graph_context":
        return ", ".join(state.get("case_ids", [])) or "no cases"

    if node == "document_context":
        return f"{len(state.get('chunks', []))} relevant passage(s)"

    return ""


def run_question(app, question):
    """
    Stream the workflow, printing each step as it finishes.
    Returns the final state.
    """

    state = {"question": question}
    remaining = [step for step in STEPS]

    console.print()

    with console.status(f"{remaining[0][2]}...") as status:

        for update in app.stream(state, stream_mode="updates"):

            for node, values in update.items():

                state.update(values or {})

                # Steps the route skips are listed as skipped
                planned = STEPS_FOR_ROUTE.get(
                    state.get("route"), {step[0] for step in STEPS}
                )

                while remaining and remaining[0][0] != node:
                    skipped = remaining.pop(0)
                    if skipped[0] not in planned:
                        console.print(
                            f"  [dim]–  {skipped[1]}: skipped, "
                            f"not needed for this question[/dim]"
                        )

                if remaining:
                    done = remaining.pop(0)
                    detail = step_detail(node, state)
                    console.print(
                        f"  [green]✓[/green]  {done[1]}"
                        + (f"  [dim]→ {detail}[/dim]" if detail else "")
                    )

                upcoming = [
                    step for step in remaining if step[0] in planned
                ]

                if upcoming:
                    status.update(f"{upcoming[0][2]}...")

    return state


# ============================================================
# OUTPUT
# ============================================================

def document_files(facts):
    """
    DOC id -> PDF file name, taken from the judgment facts.
    """

    files = {}

    for case in facts:
        for judgment in case["judgments"]:
            match = re.search(r"document (\S+) \(([^)]+)\)", judgment)
            if match:
                files[match.group(1)] = match.group(2)

    return files


def friendly_reason(reason):

    if reason.startswith("semantic match"):
        return "Closest match by meaning " + reason[len("semantic match"):]

    return "Linked to " + reason


def relevance(score):

    label = (
        "[green]High[/green]" if score >= 0.80
        else "[yellow]Medium[/yellow]" if score >= 0.70
        else "[red]Low[/red]"
    )

    return f"{label} ({score:.2f})"


def shorten(text, limit=220):

    text = " ".join(text.split())

    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def show_result(driver, state):

    route = state.get("route", "both")
    case_ids = state.get("case_ids", [])
    facts = case_facts(driver, case_ids) if case_ids else []
    files = document_files(facts)

    # 1. The answer ------------------------------------------------

    console.print()
    console.print(Panel(
        Markdown(state.get("answer", "No answer was produced.")),
        title="[bold]Answer[/bold]",
        title_align="left",
        border_style="green",
        padding=(1, 2)
    ))
    console.print(Padding(
        Text(
            "Sources in brackets: [Knowledge Graph] = structured "
            "case records; [CASE…, DOC…, page N] = a page of a "
            "judgment document.",
            style="dim"
        ),
        (0, 2)
    ))

    # 2. How it was found -----------------------------------------

    console.print()
    console.rule("[bold]How this answer was found[/bold]", style="cyan")

    title, explanation = ROUTE_TEXT.get(route, ROUTE_TEXT["both"])

    console.print()
    console.print(f"  [bold cyan]Approach:[/bold cyan] {title}")
    console.print(Padding(Text(explanation, style="dim"), (0, 2)))

    # Cases identified
    titles = {
        case["case_id"]: f"{case['title']} ({case['year']})"
        for case in facts
    }

    cases_table = Table(
        title="Cases identified",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE_HEAVY,
        expand=True
    )
    cases_table.add_column("Case", style="bold", no_wrap=True)
    cases_table.add_column("Title")
    cases_table.add_column("Why it was selected", style="dim")

    for case_id in case_ids:
        reasons = state.get("case_reasons", {}).get(case_id, [])
        cases_table.add_row(
            case_id,
            titles.get(case_id, ""),
            "\n".join(friendly_reason(reason) for reason in reasons)
        )

    console.print()

    if case_ids:
        console.print(cases_table)
    else:
        console.print("  [yellow]No matching cases were found.[/yellow]")

    # Case records (knowledge graph)
    if route in ("graph", "both") and facts:

        for case in facts:

            records = Table(
                title=f"Case record: {case['case_id']}",
                title_justify="left",
                title_style="bold",
                box=box.SIMPLE_HEAVY,
                show_header=False,
                expand=True
            )
            records.add_column("Field", style="cyan", no_wrap=True)
            records.add_column("Value")

            records.add_row("Title", f"{case['title']} ({case['year']})")

            for field, label in FACT_FIELDS:
                for value in case[field]:
                    records.add_row(label, value)

            console.print(records)

    # Judgment passages (vector search)
    chunks = state.get("chunks") or []

    if route in ("documents", "both"):

        console.print("[bold]Supporting passages from judgments[/bold]")
        console.print()

        if not chunks:
            console.print(
                "  [yellow]No judgment passages were found.[/yellow]"
            )

        # One block per passage, so long section names and
        # excerpts wrap cleanly at any terminal width
        for number, chunk in enumerate(chunks, start=1):

            case_id, document_id, page = chunk["source"].split(", ")

            console.print(
                f"  [cyan]{number}.[/cyan] "
                f"[bold]{files.get(document_id, document_id)}[/bold], "
                f"{page}  [dim]({case_id})[/dim]  "
                f"Relevance: {relevance(chunk['score'])}"
            )
            console.print(Padding(
                Text(f"Section: {chunk['section']}", style="italic"),
                (0, 0, 0, 5)
            ))
            console.print(Padding(
                Text(f"“{shorten(chunk['text'])}”", style="dim"),
                (0, 0, 1, 5)
            ))

    console.print()


def show_welcome():

    console.print(Panel(
        Group(
            Text("Litigation AI Assistant", style="bold"),
            Text(
                "Answers questions about tax litigation cases using "
                "the firm's case records and the full text of the "
                "judgments, and shows where every answer came from."
            ),
            Text(""),
            Text.from_markup(
                "Type a question, or [bold]help[/bold] for sample "
                "questions, [bold]about[/bold] for how it works, "
                "[bold]exit[/bold] to quit."
            )
        ),
        border_style="cyan",
        padding=(1, 2)
    ))


def show_help():

    table = Table(
        title="Sample questions (type the number to ask one)",
        title_justify="left",
        title_style="bold",
        box=box.SIMPLE,
        show_header=False
    )
    table.add_column(justify="right", style="cyan")
    table.add_column()

    for number, question in enumerate(SAMPLE_QUESTIONS, start=1):
        table.add_row(str(number), question)

    console.print()
    console.print(table)


def show_about():

    console.print()
    console.print(Panel(
        Markdown(f"""
**Two sources of information**

- **Case records (knowledge graph):** structured facts about
  each case: companies, tax issues, courts, tribunals, lawyers,
  judgments and related cases, stored in a Neo4j graph database.
- **Judgment documents:** the full text of the judgment PDFs,
  split into passages and indexed by meaning (vector search), so
  a passage can be found even if it uses different words.

**What happens when you ask a question**

1. **Understand the question.** The AI model works out which
   companies, cases, lawyers or courts it refers to, and decides
   whether the answer needs case records, judgment text, or both.
2. **Look up case records** for the cases identified.
3. **Search the judgment documents** of those cases for the
   most relevant passages.
4. **Write the answer** using *only* that information, citing a
   source for every statement. If something is not in the
   sources, it says so rather than guessing.

**AI model:** `{LLM_MODEL}`. It is instructed to answer only
from these two sources, not from outside knowledge.
"""),
        title="[bold]How it works[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(1, 2)
    ))


def show_error(error):

    message = str(error)

    if "rate" in message.lower() or "429" in message:
        hint = "The AI service is busy (rate limit). Wait a moment and try again."
    elif "401" in message or "api key" in message.lower():
        hint = "The AI service rejected the API key. Check GROQ_API_KEY in .env."
    else:
        hint = "Something went wrong while answering. Details below."

    console.print(Panel(
        f"{hint}\n\n[dim]{type(error).__name__}: {shorten(message, 400)}[/dim]",
        title="[bold]Could not answer[/bold]",
        title_align="left",
        border_style="red"
    ))


# ============================================================
# MAIN
# ============================================================

def ask(app, driver, question):

    try:
        state = run_question(app, question)
    except Exception as error:
        show_error(error)
        return

    show_result(driver, state)


def main():

    try:
        with console.status("Connecting to the case database..."):
            driver = get_driver()
    except Exception:
        console.print(Panel(
            "Could not connect to the case database (Neo4j).\n"
            "Start Neo4j and try again.",
            border_style="red"
        ))
        sys.exit(1)

    try:

        with console.status("Loading..."):
            app = build_workflow(driver, create_llm())
            get_model(EMBEDDING_CONFIG.model_name)

        # Question on the command line: answer it and exit
        question = " ".join(sys.argv[1:]).strip()

        if question:
            console.print(f"\n[bold]Question:[/bold] {question}")
            ask(app, driver, question)
            return

        show_welcome()

        while True:

            question = console.input(
                "\n[bold cyan]Ask a question ›[/bold cyan] "
            ).strip()

            if question.lower() in ("", "exit", "quit"):
                break

            if question.lower() == "help":
                show_help()
                continue

            if question.lower() == "about":
                show_about()
                continue

            if question.isdigit() and 1 <= int(question) <= len(SAMPLE_QUESTIONS):
                question = SAMPLE_QUESTIONS[int(question) - 1]
                console.print(f"[bold]Question:[/bold] {question}")

            ask(app, driver, question)

    finally:
        driver.close()


if __name__ == "__main__":
    main()
