from pathlib import Path

p = Path("research_analyzer.py")
s = p.read_text(encoding="utf-8")

marker = "def generate_report(research_question, source_analyses, sources):"

code = '''
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

    print("\\\\nSource evidence profiles created:", len(profiles))

    for source_id, profile in sorted(profiles.items()):
        total_items = sum(
            len(profile[category]) for category in categories
        )
        print(
            f"  {source_id}: {profile['chunk_count']} chunks, "
            f"{total_items} evidence items"
        )

    return profiles


'''

if "def build_source_evidence_profiles" in s:
    raise SystemExit("Function already exists.")

if marker not in s:
    raise SystemExit("Insertion point not found.")

p.write_text(s.replace(marker, code + marker, 1), encoding="utf-8")

print("Source evidence profile function added successfully.")