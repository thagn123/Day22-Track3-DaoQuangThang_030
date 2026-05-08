import sys

path = "generate_artifacts.py"
with open(path, encoding="utf-8") as f:
    content = f.read()

replacements = [
    ("\u2713", "[OK]"),
    ("\u26a0", "[WARN]"),
    ("\u2718", "[FAIL]"),
    ("\u279c", "-->"),
]
for old, new in replacements:
    content = content.replace(old, new)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed unicode in generate_artifacts.py")
