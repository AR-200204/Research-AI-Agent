
import json
from pathlib import Path
from datetime import datetime
import ollama

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parents[1]
REPORT_DIR = PROJECT_DIR / "reports"

MODEL_NAME = "qwen3:4b-instruct"
CONTEXT_SIZE = 2048

CHECKPOINT_FILE = REPORT_DIR / "analysis_progress_research_20261118_062142.json"
EVIDENCE_FILE = REPORT_DIR / "evidence_research_20261118_062142.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    temp_path.replace(path)


def extract_evidence(chunk_id, source_id, chunk_text):

    prompt = f"""
You are extracting evidence from an AI-generated analysis.

SOURCE ID: {source_id}
CHUNK ID: {chunk_id}

ANALYSIS:
{chunk_text}

Extract only information explicitly present.

Return valid JSON:
{{
  "main_findings": [],
  "methods_and_technologies": [],
  "datasets": [],
  "limitations": [],
  "author_reported_gaps": [],
  "possible_inferred_gaps": [],
  "uncertainties": []
}}

Rules:
- Never invent information.
- Do not invent research gaps.
- Use empty lists when information is unavailable.
- Preserve uncertainty and incomplete statements.
- These are AI-derived summaries, not direct quotations.
- Return JSON only.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n/no_think"
                }
            ],
            format="json",
            options={
                "temperature": 0.1,
                "num_ctx": CONTEXT_SIZE,
                "num_predict": 600
            }
        )

        result = json.loads(response.message.content.strip())

        required_keys = [
            "main_findings",
            "methods_and_technologies",
            "datasets",
            "limitations",
            "author_reported_gaps",
            "possible_inferred_gaps",
            "uncertainties"
        ]

        for key in required_keys:
            if not isinstance(result.get(key), list):
                result[key] = []

        return {
            "source_id": source_id,
            "chunk_id": chunk_id,
            "evidence_origin": "AI-generated analysis of extracted source content",
            "extracted_at": datetime.now().isoformat(timespec="seconds"),
            **result
        }

    except Exception as error:
        print(f"Extraction error for {chunk_id}: {error}")
        return None


def main():

    print("=" * 65)
    print("OPENRESEARCH AI - EVIDENCE EXTRACTOR")
    print("=" * 65)

    if not CHECKPOINT_FILE.exists():
        print("Analysis checkpoint not found:", CHECKPOINT_FILE)
        return

    checkpoint = load_json(CHECKPOINT_FILE)
    chunks = checkpoint.get("completed_chunks", {})

    if not chunks:
        print("No completed chunks found.")
        return

    if EVIDENCE_FILE.exists():
        evidence_data = load_json(EVIDENCE_FILE)
    else:
        evidence_data = {
            "project": "OpenResearch AI",
            "model": MODEL_NAME,
            "evidence_records": {}
        }

    records = evidence_data.setdefault("evidence_records", {})

    print("Total available chunks:", len(chunks))
    print("Already extracted:", len(records))

    for chunk_id, chunk_text in chunks.items():

        if chunk_id in records:
            print("Skipping:", chunk_id)
            continue

        source_id = chunk_id.rsplit("_chunk_", 1)[0]

        print("\nExtracting:", chunk_id)

        record = extract_evidence(
            chunk_id,
            source_id,
            str(chunk_text)
        )

        if record is None:
            print("Failed. It can be retried later.")
            continue

        records[chunk_id] = record
        save_json(EVIDENCE_FILE, evidence_data)

        print("Saved evidence for:", chunk_id)

    successful = len(records)
    failed = len(chunks) - successful

    print("\nEVIDENCE EXTRACTION FINISHED")
    print("Successful records:", successful)
    print("Remaining records:", failed)
    print("Output:", EVIDENCE_FILE)


if __name__ == "__main__":
    main()
