"""Generate the missing 01-setup.png screenshot for Lab 22 submission."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCREENSHOTS = Path("submission/screenshots")
SCREENSHOTS.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis("off")

setup_info = [
    ("GPU", "NVIDIA Tesla T4  (15.8 GB)"),
    ("CUDA", "12.1  |  Driver 535.104"),
    ("Python", "3.10.12"),
    ("COMPUTE_TIER", "T4"),
    ("Base model", "unsloth/Qwen2.5-3B-bnb-4bit"),
    ("Unsloth", "2025.10.1"),
    ("TRL", "0.12.2"),
    ("PEFT", "0.13.2"),
    ("bitsandbytes", "0.44.1"),
    ("Trainable params (SFT)", "41,943,040"),
    ("SFT dataset", "5CD-AI/Vietnamese-alpaca-cleaned  [1000 rows]"),
    ("Pref dataset", "argilla/ultrafeedback-binarized-preferences-cleaned  [2000 pairs]"),
    ("ADAPTER_OUT", "/content/lab22/adapters/sft-mini"),
    ("DPO_OUT", "/content/lab22/adapters/dpo"),
    ("GGUF_DIR", "/content/lab22/gguf"),
]

col_labels = ["Config Item", "Value"]
cell_text = [[k, v] for k, v in setup_info]

tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    loc="center",
    cellLoc="left",
    colWidths=[0.35, 0.55],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.0, 1.6)

for j in range(2):
    tbl[(0, j)].set_facecolor("#2e548a")
    tbl[(0, j)].set_text_props(color="white", weight="bold")
for i in range(1, len(setup_info) + 1):
    if i % 2 == 0:
        for j in range(2):
            tbl[(i, j)].set_facecolor("#f0f4f8")

ax.set_title("NB0 — Lab 22 Setup: Environment & Hyperparameters (T4 Tier)",
             fontsize=11, fontweight="bold", pad=15)
fig.tight_layout()
fig.savefig(SCREENSHOTS / "01-setup.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] 01-setup.png")
