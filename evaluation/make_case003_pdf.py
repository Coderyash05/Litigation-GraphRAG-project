"""
Generate a synthetic judgment PDF for CASE003 (DEF Industries),
matching the case already present in the Neo4j graph:

    CASE003 | DEF Industries vs Income Tax Department | 2022
    TaxIssue: Depreciation Claim Dispute | Section 32

It deliberately uses the same vocabulary as CASE001 (evidence,
invoices, arguments, court's consideration), so the evaluation
can detect chunks retrieved from the wrong case.

Run from the project root:

    python evaluation/make_case003_pdf.py
"""

from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate


OUTPUT_PATH = Path("documents") / "CASE003_Judgment.pdf"

PAGES = [
    [
        ("Title", "INCOME TAX LITIGATION — DUMMY JUDGMENT"),
        ("Heading2", "CASE003: DEF Industries vs Income Tax Department"),
        ("Heading3", "Background"),
        ("Body",
         "DEF Industries, a textile manufacturer, installed two new "
         "weaving machines during the financial year 2022 and claimed "
         "depreciation on both machines at the full rate. During "
         "assessment, the Assessing Officer restricted the claim to half "
         "the normal rate, holding that the machines had not been used "
         "for the purposes of the business for at least 180 days."),
        ("Heading3", "Issue"),
        ("Body",
         "The principal issue was whether the weaving machines had been "
         "put to use for the purposes of the business during the year, "
         "and whether depreciation was allowable under Section 32 of the "
         "Income Tax Act at the rate claimed by the assessee."),
    ],
    [
        ("Heading3", "Arguments of the Assessee"),
        ("Body",
         "DEF Industries submitted installation certificates from the "
         "machine supplier, electricity consumption records, daily "
         "production logs and purchase invoices. The company contended "
         "that these records proved the first machine was commissioned "
         "in April and operated continuously thereafter."),
        ("Heading3", "Arguments of the Revenue"),
        ("Body",
         "The Revenue argued that the production logs reflected only "
         "trial runs and not commercial production. It maintained that "
         "a machine undergoing trials cannot be treated as put to use, "
         "and that the second machine had not been installed before the "
         "end of the financial year."),
    ],
    [
        ("Heading3", "Tribunal's Consideration"),
        ("Body",
         "The Tribunal examined the commissioning reports, the "
         "electricity bills and the production logs. It observed that "
         "the electricity consumption of the first machine was "
         "consistent with regular operation rather than intermittent "
         "trials."),
        ("Heading3", "Interpretation of 'Put to Use'"),
        ("Body",
         "The Tribunal held that an asset is put to use under Section 32 "
         "when it is installed and ready for use in the ordinary course "
         "of business, even if it is not used at full capacity. A machine "
         "that has not been installed, however, cannot be treated as "
         "ready for use."),
    ],
    [
        ("Heading3", "Decision"),
        ("Body",
         "The appeal of DEF Industries was partly allowed. Depreciation "
         "at the full rate was allowed on the first weaving machine. The "
         "claim on the second machine was rejected because it was "
         "installed only after the close of the financial year."),
        ("Heading3", "Outcome"),
        ("Body",
         "The matter was decided partly in favour of the assessee. The "
         "order was pronounced on 12 October 2022 by the Income Tax "
         "Appellate Tribunal, Mumbai Bench."),
    ],
]


def main():

    styles = getSampleStyleSheet()

    style_names = {
        "Title": "Title",
        "Heading2": "Heading2",
        "Heading3": "Heading3",
        "Body": "BodyText"
    }

    story = []

    for page_index, page in enumerate(PAGES):

        if page_index > 0:
            story.append(PageBreak())

        for style, text in page:
            story.append(Paragraph(text, styles[style_names[style]]))

    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    SimpleDocTemplate(str(OUTPUT_PATH), pagesize=A4).build(story)

    print(f"Created {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
