from pathlib import Path

p = Path("research_analyzer.py")
s = p.read_text(encoding="utf-8")

marker = "def generate_report(research_question, source_analyses, sources):"

code = '''

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

    return "\\\\n".join(selected)


'''

if "def select_report_evidence(" in s:
    raise SystemExit("Evidence selector already exists.")

if marker not in s:
    raise SystemExit("Insertion point not found.")

p.write_text(s.replace(marker, code + marker, 1), encoding="utf-8")

print("Evidence selector added successfully.")