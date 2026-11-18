
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
EVIDENCE_FILE = REPORT_DIR / "evidence_research_20261118_062142.json"

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
# DETAILED OLLAMA RESPONSE
# --------------------------------------------------

def ask_ollama_detailed(prompt, max_tokens=450):

    try:
        options = OPTIONS.copy()
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

        return {
            "content": response.message.content.strip(),
            "done": response.done,
            "done_reason": response.done_reason,
            "prompt_eval_count": response.prompt_eval_count,
            "eval_count": response.eval_count
        }

    except Exception as error:
        print("\nDetailed Ollama error:")
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

def load_evidence_by_source():

    required_fields = [
        "main_findings",
        "methods_and_technologies",
        "datasets",
        "limitations",
        "author_reported_gaps",
        "possible_inferred_gaps",
        "uncertainties"
    ]

    if not EVIDENCE_FILE.exists():
        raise FileNotFoundError(f"Evidence file not found: {EVIDENCE_FILE}")

    with open(EVIDENCE_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    records = data.get("evidence_records", {})

    if not records:
        raise ValueError("Evidence file contains no records.")

    grouped = {}

    for chunk_id, record in records.items():
        if not isinstance(record, dict):
            raise ValueError(f"Invalid evidence record: {chunk_id}")

        for field in required_fields:
            if not isinstance(record.get(field), list):
                raise ValueError(
                    f"Invalid field {field} in {chunk_id}"
                )

        source_id = record.get("source_id")

        if not source_id:
            raise ValueError(f"Missing source ID in {chunk_id}")

        grouped.setdefault(source_id, []).append(record)

    print("\\nEvidence records loaded:", len(records))
    print("Sources with evidence:", len(grouped))

    for source_id, source_records in sorted(grouped.items()):
        print(f"  {source_id}: {len(source_records)} chunks")

    return grouped



def build_source_evidence_profiles(grouped_evidence):
    categories = [
        "main_findings",
        "methods_and_technologies",
        "datasets",
        "limitations",
        "author_reported_gaps",
        "possible_inferred_gaps",
        "uncertainties"
    ]

    profiles = {}

    for source_id, records in grouped_evidence.items():
        profile = {
            "source_id": source_id,
            "chunk_count": len(records)
        }

        for category in categories:
            profile[category] = []

            for record in records:
                for item in record.get(category, []):
                    profile[category].append({
                        "chunk_id": record.get("chunk_id"),
                        "evidence_origin": record.get("evidence_origin"),
                        "text": item
                    })

        profiles[source_id] = profile

    print("\\nSource evidence profiles created:", len(profiles))

    for source_id, profile in sorted(profiles.items()):
        total_items = sum(
            len(profile[category]) for category in categories
        )
        print(
            f"  {source_id}: {profile['chunk_count']} chunks, "
            f"{total_items} evidence items"
        )

    return profiles




def select_report_evidence(profiles, categories, max_chars=3500):

    selected = []
    used_chars = 0

    source_ids = sorted(profiles.keys())

    # Round-robin selection prevents large sources
    # from dominating the evidence selection.
    source_items = {}

    for source_id in source_ids:
        items = []

        for category in categories:
            for evidence in profiles[source_id].get(category, []):
                items.append({
                    "source_id": source_id,
                    "chunk_id": evidence["chunk_id"],
                    "category": category,
                    "text": evidence["text"]
                })

        source_items[source_id] = items

    positions = {source_id: 0 for source_id in source_ids}

    while used_chars < max_chars:

        added_item = False

        for source_id in source_ids:

            position = positions[source_id]
            items = source_items[source_id]

            if position >= len(items):
                continue

            item = items[position]
            positions[source_id] += 1

            text = str(item["text"]).strip()

            formatted = (
                f"[{item['source_id']} | "
                f"{item['chunk_id']} | "
                f"{item['category']}] {text}"
            )

            if used_chars + len(formatted) > max_chars:
                continue

            selected.append(formatted)
            used_chars += len(formatted)
            added_item = True

        if not added_item:
            break

    print(
        f"Selected {len(selected)} evidence items "
        f"({used_chars}/{max_chars} characters)"
    )

    return "\n".join(selected)



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
            2200
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

        for attempt, token_budget in enumerate([600, 700, 800], 1):

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



