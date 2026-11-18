
from pathlib import Path

p = Path("research_analyzer.py")
source = p.read_text(encoding="utf-8")

start = source.index("def generate_report(")
end = source.index("# SAVE FINAL REPORT", start)

replacement = r'''
def generate_report(research_question, source_analyses, sources):

    print("\n" + "=" * 70)
    print("GENERATING EVIDENCE-BASED RESEARCH REPORT")
    print("=" * 70)

    # Load the validated evidence records.
    grouped_evidence = load_evidence_by_source()
    profiles = build_source_evidence_profiles(grouped_evidence)

    if not profiles:
        print("ERROR: No evidence profiles available.")
        return None

    # Build source metadata for accurate source descriptions.
    source_metadata = {
        source["source_id"]: source
        for source in sources
        if source.get("source_id")
    }

    podcast_ids = []

    for source_id in profiles:
        metadata = source_metadata.get(source_id, {})
        source_type = str(metadata.get("source_type", "")).lower()

        if "podcast" in source_type:
            podcast_ids.append(source_id)

    source_catalog = []

    for source_id in sorted(profiles):
        metadata = source_metadata.get(source_id, {})

        source_catalog.append(
            f"[{source_id}] "
            f"Title: {metadata.get('title', 'Unknown')} | "
            f"Type: {metadata.get('source_type', 'Unknown')} | "
            f"Status: {metadata.get('extraction_status', 'Unknown')}"
        )

    source_catalog_text = "\n".join(source_catalog)

    if podcast_ids:
        print("Contextual podcast sources:", podcast_ids)

    # Each section receives only the relevant evidence categories.
    report_sections = [
        (
            "# Research Overview",
            "Introduce the topic, its importance, scope and major themes.",
            ["main_findings"],
            2200
        ),
        (
            "# Key Findings",
            "Present the main findings, identify common themes and "
            "compare findings across academic sources.",
            ["main_findings"],
            2800
        ),
        (
            "# Methods and Technologies",
            "Explain algorithms, tools, techniques and datasets. "
            "Do not invent technical details.",
            ["methods_and_technologies", "datasets"],
            2600
        ),
        (
            "# Challenges and Limitations",
            "Discuss limitations, uncertainties and challenges "
            "actually supported by the evidence.",
            ["limitations", "uncertainties"],
            2200
        ),
        (
            "# Research Gaps and Future Directions",
            "Identify explicitly documented research gaps if present. "
            "Then carefully derive possible future directions from "
            "the findings and limitations. Clearly label these as "
            "your own evidence-based inferences, not author-reported gaps.",
            [
                "author_reported_gaps",
                "possible_inferred_gaps",
                "main_findings",
                "limitations",
                "uncertainties"
            ],
            2800
        ),
        (
            "# Conclusion",
            "Answer the research question using the strongest available "
            "findings and methods. Acknowledge evidence limitations.",
            ["main_findings", "methods_and_technologies"],
            2000
        )
    ]

    generated_sections = []

    for index, (heading, instructions, categories, max_chars) in enumerate(
        report_sections, 1
    ):

        print("\n" + "=" * 60)
        print(f"SECTION {index}/{len(report_sections)}")
        print(heading)

        evidence = select_report_evidence(
            profiles,
            categories,
            max_chars=max_chars
        )

        if not evidence.strip():
            print("WARNING: No evidence selected for this section.")
            evidence = "No relevant evidence was extracted for this section."

        prompt = f"""
You are an academic literature review writer.

RESEARCH QUESTION:
{research_question}

SOURCE CATALOG:
{source_catalog_text}

SELECTED EVIDENCE:
{evidence}

IMPORTANT EVIDENCE PROVENANCE:
The evidence items are AI-generated summaries of extracted source
content. They are not direct quotations from the original papers.
Their source IDs and chunk IDs provide traceability to the extraction
pipeline, but not exact page-level citations.

SOURCE HANDLING:
1. Use source IDs such as [S001] to support claims.
2. Podcast sources are contextual material, not peer-reviewed
   academic evidence. Do not use them as the sole support for
   scientific or academic claims.
3. Do not use sources that are absent from the selected evidence.
4. Do not invent authors, publication years, statistics or references.

SECTION:
{heading}

TASK:
{instructions}

REQUIREMENTS:
1. Use only the supplied evidence.
2. Compare academic sources where possible.
3. Never invent findings, methods, datasets or statistics.
4. Distinguish source-reported findings from your own interpretation.
5. If evidence is insufficient, explicitly acknowledge that.
6. Do not claim that authors reported research gaps when the
   author-reported gap evidence is empty.
7. Any inferred research gap must be labelled as an inference.
8. Use clear academic language and complete paragraphs.
9. Include relevant source IDs after supported claims.
10. Do not repeat the section heading.
11. Return only the section content.
"""

        result = None

        for attempt, token_budget in enumerate([450, 600], 1):

            print(
                f"Generation attempt {attempt}, "
                f"token budget: {token_budget}"
            )

            result = ask_ollama_detailed(
                prompt,
                max_tokens=token_budget
            )

            if result is None:
                print("Ollama request failed.")
                return None

            reason = result["done_reason"]
            content = result["content"]

            print("Completion reason:", reason)
            print("Generated tokens:", result["eval_count"])

            if not content:
                print("Empty response. Retrying...")
                continue

            if reason == "length" or not result["done"]:
                print("Token limit reached. Retrying...")
                continue

            break

        else:
            print("ERROR: Could not complete section:", heading)
            return None

        heading_text = heading.lstrip("#").strip().lower()

        cleaned_lines = [
            line for line in content.splitlines()
            if line.strip().lstrip("#").strip().lower()
            != heading_text
        ]

        content = "\n".join(cleaned_lines).strip()

        if not content:
            print("ERROR: Section contains no usable content.")
            return None

        generated_sections.append(
            f"{heading}\n\n{content}"
        )

        print("Section completed successfully.")

    final_report = "\n\n".join(generated_sections)

    missing_sections = [
        heading
        for heading, _, _, _ in report_sections
        if heading not in final_report
    ]

    if missing_sections:
        print("ERROR: Missing sections:", missing_sections)
        return None

    print("\nAll six evidence-based sections completed.")

    return final_report

'''

source = source[:start] + replacement + "\n\n" + source[end:]

p.write_text(source, encoding="utf-8")

print("generate_report() replaced successfully.")
