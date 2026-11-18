
import json
from pathlib import Path
from datetime import datetime

import ollama


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parents[1]

RESEARCH_DIR = PROJECT_DIR / "research_data"
REPORT_DIR = PROJECT_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "qwen3:4b-instruct"

# Keep chunks small for the 2048-token context window.
MAX_CHUNK_CHARS = 2400

OPTIONS = {
    "temperature": 0.2,
    "num_ctx": 2048,
    "num_predict": 180
}


# --------------------------------------------------
# FIND LATEST RESEARCH FILE
# --------------------------------------------------

def get_latest_research_file():

    files = list(RESEARCH_DIR.glob("research_*.json"))

    if not files:
        print("No research JSON files found.")
        return None

    latest_file = max(files, key=lambda file: file.stat().st_mtime)

    print(f"\nLatest research file: {latest_file.name}")

    return latest_file


# --------------------------------------------------
# LOAD RESEARCH DATA
# --------------------------------------------------

def load_research(file_path):

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        print("\nResearch data loaded successfully.")
        print("Research question:", data.get("research_question"))
        print("Total sources:", data.get("total_sources", 0))

        return data

    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not load research data: {error}")
        return None


# --------------------------------------------------
# SPLIT LONG TEXT INTO SMALL CHUNKS
# --------------------------------------------------

def split_text(text, max_chars=MAX_CHUNK_CHARS):

    chunks = []

    for start in range(0, len(text), max_chars):
        chunk = text[start:start + max_chars]

        if chunk.strip():
            chunks.append(chunk)

    return chunks


# --------------------------------------------------
# CALL LOCAL OLLAMA MODEL
# --------------------------------------------------

def ask_ollama(prompt, max_tokens=None):

    try:
        options = OPTIONS.copy()

        if max_tokens:
            options["num_predict"] = max_tokens

        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n/no_think"
                }
            ],
            options=options
        )

        return response.message.content.strip()

    except Exception as error:
        print("\nOllama error:")
        print(f"{type(error).__name__}: {error}")
        return None


# --------------------------------------------------
# CHECKPOINT MANAGEMENT
# --------------------------------------------------

def save_checkpoint(checkpoint_path, checkpoint_data):

    with open(checkpoint_path, "w", encoding="utf-8") as file:
        json.dump(
            checkpoint_data,
            file,
            indent=4,
            ensure_ascii=False
        )


def load_checkpoint(checkpoint_path):

    if checkpoint_path.exists():
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as file:
                data = json.load(file)

            print("\nExisting analysis checkpoint found.")
            print("Completed chunks will be reused.")

            return data

        except (OSError, json.JSONDecodeError):
            print("Checkpoint could not be read. Starting fresh.")

    return {
        "completed_chunks": {},
        "source_analyses": {}
    }


# --------------------------------------------------
# ANALYZE ONE SOURCE CHUNK
# --------------------------------------------------

def analyze_chunk(source, chunk, chunk_number, total_chunks):

    prompt = f"""
You are a careful academic research assistant.

Analyze the following extract from a research source.

SOURCE ID: {source['source_id']}
SOURCE TITLE: {source['title']}
SOURCE TYPE: {source['source_type']}
CHUNK: {chunk_number} of {total_chunks}

SOURCE CONTENT:
{chunk}

Instructions:

1. Identify the main ideas and findings.
2. Identify methods, algorithms, or datasets mentioned.
3. Identify limitations or challenges mentioned.
4. Include important technical details.
5. Only report information supported by this extract.
6. Do not invent facts, statistics, authors, or conclusions.
7. If something is not mentioned, do not guess.
8. Keep the response concise.

Return your analysis using these headings:

MAIN IDEAS:
METHODS AND TECHNIQUES:
FINDINGS:
LIMITATIONS:
IMPORTANT DETAILS:

Use the source ID {source['source_id']} when referring to this evidence.
"""

    return ask_ollama(prompt, max_tokens=180)


# --------------------------------------------------
# ANALYZE ALL SOURCES
# --------------------------------------------------

def analyze_sources(sources, checkpoint_path):

    checkpoint = load_checkpoint(checkpoint_path)

    completed_chunks = checkpoint["completed_chunks"]
    source_analyses = checkpoint["source_analyses"]

    eligible_sources = [
        source for source in sources
        if source.get("extraction_status") in ("success", "snippet_only")
        and source.get("content", "").strip()
    ]

    skipped_sources = [
        source for source in sources
        if source not in eligible_sources
    ]

    print("\nSources available for AI analysis:", len(eligible_sources))
    print("Sources skipped:", len(skipped_sources))

    for source in eligible_sources:

        source_id = source["source_id"]

        print("\n" + "=" * 70)
        print(f"ANALYZING {source_id}")
        print("Title:", source["title"])
        print("=" * 70)

        chunks = split_text(source["content"])

        print("Number of chunks:", len(chunks))

        chunk_results = []

        for index, chunk in enumerate(chunks, start=1):

            chunk_key = f"{source_id}_chunk_{index}"

            if chunk_key in completed_chunks:

                print(f"\nChunk {index}/{len(chunks)} already analyzed.")
                result = completed_chunks[chunk_key]

            else:

                print(f"\nAnalyzing chunk {index}/{len(chunks)}...")

                result = analyze_chunk(
                    source,
                    chunk,
                    index,
                    len(chunks)
                )

                if not result:
                    print("Chunk analysis failed. You can rerun to retry.")
                    continue

                completed_chunks[chunk_key] = result

                save_checkpoint(
                    checkpoint_path,
                    checkpoint
                )

                print("Chunk analysis completed.")

            chunk_results.append(result)

        if chunk_results:

            source_analyses[source_id] = {
                "title": source["title"],
                "source_type": source["source_type"],
                "extraction_status": source["extraction_status"],
                "analysis": "\n\n".join(chunk_results)
            }

            save_checkpoint(
                checkpoint_path,
                checkpoint
            )

    return source_analyses, skipped_sources


# --------------------------------------------------
# GENERATE FINAL RESEARCH REPORT
# --------------------------------------------------

def generate_report(research_question, source_analyses, sources):

    print("\n" + "=" * 70)
    print("GENERATING FINAL RESEARCH REPORT")
    print("=" * 70)

    evidence_sections = []

    for source_id, analysis in source_analyses.items():

        source_text = str(analysis.get("analysis", "")).strip()

        # Preserve more evidence from each source.
        compact_analysis = source_text[:1100]

        evidence_sections.append(
            f"""
SOURCE ID: {source_id}
TITLE: {analysis.get('title', 'Unknown title')}
SOURCE TYPE: {analysis.get('source_type', 'Unknown')}

ANALYSIS:
{compact_analysis}
"""
        )

    evidence = "\n".join(evidence_sections)

    report_sections = [
        (
            "# Research Overview",
            "Introduce the research topic, its importance, scope, and "
            "the main themes emerging from the collected evidence."
        ),
        (
            "# Key Findings",
            "Present the most important findings across sources. "
            "Compare related findings and identify agreements or "
            "differences where evidence permits."
        ),
        (
            "# Methods and Technologies",
            "Explain the methods, algorithms, tools, datasets, and "
            "technologies discussed in the sources."
        ),
        (
            "# Challenges and Limitations",
            "Discuss technical, methodological, and scientific "
            "limitations supported by the evidence."
        ),
        (
            "# Research Gaps and Future Directions",
            "Identify gaps explicitly supported by the sources. "
            "Clearly distinguish source-reported gaps from reasonable "
            "future research suggestions."
        ),
        (
            "# Conclusion",
            "Summarize the overall evidence and answer the research "
            "question without introducing unsupported claims."
        )
    ]

    generated_sections = []

    for index, (heading, instructions) in enumerate(report_sections, 1):

        print(f"\nGenerating section {index}/{len(report_sections)}...")
        print(heading)

        prompt = f"""
You are an academic literature review writer.

RESEARCH QUESTION:
{research_question}

AVAILABLE SOURCE EVIDENCE:
{evidence}

TASK:
Write the following report section:

{heading}

Instructions:
{instructions}

IMPORTANT RULES:
1. Use only the supplied source evidence.
2. Cite supporting claims using IDs such as [S001].
3. Never invent facts, statistics, experiments, or references.
4. Do not cite sources that do not support the claim.
5. Compare sources when evidence allows.
6. Distinguish established findings from suggestions.
7. If evidence is insufficient, explicitly acknowledge this.
8. Use clear academic language.
9. Avoid repetition.
10. Write complete sentences and finish the section properly.
11. Do not repeat the section heading.
12. Return only the section content.
"""

        result = ask_ollama(prompt, max_tokens=280)

        # Retry if the section appears empty or incomplete.
        if (
            not result
            or len(result.strip()) < 80
            or result.strip()[-1] not in ".!?)]}\"'"
        ):

            print("Section may be incomplete. Retrying...")

            result = ask_ollama(prompt, max_tokens=350)

        if not result:
            result = (
                "Section generation failed. "
                "Further analysis is required."
            )

        generated_sections.append(
            f"{heading}\n\n{result.strip()}"
        )

        print("Section completed.")

    final_report = "\n\n".join(generated_sections)

    # Verify that all required sections exist.
    missing_sections = [
        heading
        for heading, _ in report_sections
        if heading not in final_report
    ]

    if missing_sections:
        print("WARNING: Missing report sections:", missing_sections)

    else:
        print("\nAll report sections generated successfully.")

    return final_report

# SAVE FINAL REPORT
# --------------------------------------------------

def save_report(
    research_question,
    report_content,
    source_analyses,
    skipped_sources,
    sources
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = REPORT_DIR / f"research_report_{timestamp}.md"

    with open(report_path, "w", encoding="utf-8") as file:

        file.write("# OpenResearch AI - Research Report\n\n")

        file.write(f"**Research question:** {research_question}\n\n")
        file.write(
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        )

        file.write("---\n\n")

        file.write(report_content)

        file.write("\n\n---\n\n")
        file.write("# References and Source Information\n\n")

        for source in sources:

            source_id = source["source_id"]

            file.write(f"### [{source_id}] {source['title']}\n\n")

            file.write(f"- Type: {source['source_type']}\n")
            file.write(f"- URL: {source['url']}\n")
            file.write(
                f"- Extraction status: {source['extraction_status']}\n"
            )

            if source.get("authors"):
                file.write(
                    f"- Authors: {', '.join(source['authors'])}\n"
                )

            if source.get("year"):
                file.write(f"- Year: {source['year']}\n")

            if source.get("doi"):
                file.write(f"- DOI: {source['doi']}\n")

            if source_id in source_analyses:
                file.write("- AI analysis: Completed\n")
            else:
                file.write("- AI analysis: Not available\n")

            file.write("\n")

        if skipped_sources:
            file.write("# Sources Not Analyzed\n\n")

            for source in skipped_sources:
                file.write(
                    f"- [{source['source_id']}] {source['title']} "
                    f"({source['extraction_status']})\n"
                )

    print("\n" + "=" * 70)
    print("RESEARCH REPORT SAVED")
    print("=" * 70)
    print(f"Report: {report_path}")

    return report_path


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def run_analyzer():

    research_file = get_latest_research_file()

    if not research_file:
        return

    research_data = load_research(research_file)

    if not research_data:
        return

    research_question = research_data.get("research_question", "")
    sources = research_data.get("sources", [])

    if not research_question or not sources:
        print("Research question or sources are missing.")
        return

    checkpoint_path = REPORT_DIR / (
        f"analysis_progress_{research_file.stem}.json"
    )

    source_analyses, skipped_sources = analyze_sources(
        sources,
        checkpoint_path
    )

    if not source_analyses:
        print("\nNo sources were successfully analyzed.")
        return

    report_content = generate_report(
        research_question,
        source_analyses,
        sources
    )

    if not report_content:
        print("\nFinal report generation failed.")
        print("Your source analyses are saved in the checkpoint.")
        return

    save_report(
        research_question,
        report_content,
        source_analyses,
        skipped_sources,
        sources
    )

    print("\nAI research analysis completed successfully!")


if __name__ == "__main__":

    print("=" * 60)
    print("WELCOME TO OPENRESEARCH AI")
    print("AI RESEARCH ANALYZER")
    print("=" * 60)

    run_analyzer()

