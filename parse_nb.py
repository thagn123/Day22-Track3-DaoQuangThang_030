"""Parse the executed notebook and extract all text outputs."""
import json
import sys

with open("Lab22_DPO_T4.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print("=" * 80)

for i, cell in enumerate(nb["cells"]):
    outputs = cell.get("outputs", [])
    if not outputs:
        continue
    
    src_lines = cell.get("source", [])
    src_preview = "".join(src_lines)[:120].replace("\n", " | ")
    print(f"\n--- Cell {i} (code) ---")
    print(f"SOURCE: {src_preview}")
    print(f"OUTPUTS ({len(outputs)}):")
    
    for j, out in enumerate(outputs):
        otype = out.get("output_type", "?")
        if otype == "stream":
            text = "".join(out.get("text", []))
            print(f"  [{otype}] {text[:500]}")
        elif otype in ("execute_result", "display_data"):
            data = out.get("data", {})
            if "text/plain" in data:
                txt = "".join(data["text/plain"])
                print(f"  [{otype}] {txt[:500]}")
            if "image/png" in data:
                print(f"  [{otype}] <IMAGE/PNG present>")
        elif otype == "error":
            ename = out.get("ename", "")
            evalue = out.get("evalue", "")
            print(f"  [ERROR] {ename}: {evalue[:200]}")
        else:
            print(f"  [{otype}]")
