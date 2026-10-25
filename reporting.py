

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


# Generate PDF report
def generate_pdf_report(
    scores,
    transcript,
    output_file="interview_report.pdf",
    tips=None,
):

    chart_file = "score_chart.png"

    labels = list(scores.keys())
    values = [
        value if value is not None else 0
        for value in scores.values()
    ]

    plt.figure(figsize=(7, 3.5))
    plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel("Score out of 100")
    plt.title("Mock Interview Performance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(chart_file)
    plt.close()

    document = SimpleDocTemplate(
        output_file,
        pagesize=A4
    )

    styles = getSampleStyleSheet()
    content = []

    content.append(
        Paragraph("InterviewSense Pro", styles["Title"])
    )

    content.append(
        Paragraph("Mock Interview Performance Report", styles["Heading2"])
    )

    content.append(Spacer(1, 15))

    score_table = [
        ["Performance Area", "Score"]
    ]

    for label, value in scores.items():
        if value is None:
            score_text = "N/A"
        else:
            score_text = str(value)
        score_table.append([label, score_text])

    table = Table(score_table, colWidths=[250, 100])

    table.setStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8)
    ])

    content.append(table)
    content.append(Spacer(1, 15))

    content.append(Image(chart_file, width=450, height=225))
    content.append(Spacer(1, 15))

    content.append(
        Paragraph("How to Improve", styles["Heading2"])
    )

    if tips:
        for tip in tips:
            content.append(
                Paragraph("- " + tip, styles["BodyText"])
            )
    else:
        content.append(
            Paragraph(
                "No major improvement areas were detected.",
                styles["BodyText"]
            )
        )

    content.append(Spacer(1, 15))

    content.append(
        Paragraph("Answer Transcripts", styles["Heading2"])
    )

    for number, answer in enumerate(transcript, start=1):
        content.append(
            Paragraph(
                "Question " + str(number) + ": " + answer["question"],
                styles["Heading3"]
            )
        )

        content.append(
            Paragraph(
                answer["answer"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 10))

    document.build(content)

    return output_file
