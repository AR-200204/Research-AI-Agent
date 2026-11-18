from pathlib import Path

p = Path("research_analyzer.py")
s = p.read_text(encoding="utf-8")

old_retry = "enumerate([450, 600], 1)"
new_retry = "enumerate([600, 700, 800], 1)"

if old_retry not in s:
    raise SystemExit("Could not find original retry settings.")

s = s.replace(old_retry, new_retry, 1)

old_limit = '''            "Present the main findings, identify common themes and "
            "compare findings across academic sources.",
            ["main_findings"],
            2800'''

new_limit = '''            "Present the main findings, identify common themes and "
            "compare findings across academic sources.",
            ["main_findings"],
            2200'''

if old_limit not in s:
    raise SystemExit("Could not find Key Findings evidence limit.")

s = s.replace(old_limit, new_limit, 1)

p.write_text(s, encoding="utf-8")

print("Report generation settings updated.")