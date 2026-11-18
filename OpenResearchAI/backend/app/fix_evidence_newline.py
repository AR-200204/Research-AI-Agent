from pathlib import Path

p = Path("research_analyzer.py")
s = p.read_text(encoding="utf-8")

lines = s.splitlines()

for i, line in enumerate(lines):
    if "join(selected)" in line:
        lines[i] = '    return "\\n".join(selected)'
        break
else:
    raise SystemExit("Could not find join(selected)")

p.write_text("\n".join(lines) + "\n", encoding="utf-8")

print("Evidence newline formatting fixed.")