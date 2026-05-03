"""Side-by-side comparison plot: LLP-mean sweep vs VGGSound-mean sweep."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jaemo/AVVP_stage12_clean/results")
OUT_DIR = ROOT / "mean_source_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)
AV2A_METRICS_PATH = Path(
    "/home/jaemo/AV2A_pristine/runs/llp_clipclap_20260420_l2norm_full/per_class_metrics.json"
)


def load_main(p: Path) -> list[dict]:
    rs = json.load(open(p))
    return sorted([r for r in rs if r["is_main"]], key=lambda r: r["lambda"])


# Auto-discover all sweep result files; LLP self-mean is the reference.
SOURCES = [
    ("LLP test mean (transductive)",     ROOT / "sweep_lambda" / "sweep_results.json",            "#1f77b4", "-o"),
    ("VGGSound mean (N=3008, batch_1)",   ROOT / "sweep_lambda_vggsound" / "sweep_results.json",   "#d62728", "-s"),
    ("AudioCaps mean (N=3000)",           ROOT / "sweep_lambda_audiocaps" / "sweep_results.json",  "#2ca02c", "-^"),
    ("FSD50K mean (N=36796)",             ROOT / "sweep_lambda_fsd50k" / "sweep_results.json",     "#ff7f0e", "-D"),
    ("DCASE2017 mean (N=2490)",           ROOT / "sweep_lambda_dcase2017" / "sweep_results.json",  "#9467bd", "-v"),
]
sources_ready = [(name, load_main(p), color, marker) for (name, p, color, marker) in SOURCES if p.exists()]
if not sources_ready:
    raise SystemExit("no sweep_results.json found yet")

lams = [r["lambda"] for r in sources_ready[0][1]]


def col(rs, k): return [r[k] for r in rs]


def as_fraction(x):
    x = float(x)
    return x / 100.0 if x > 1.0 else x


def load_av2a_baseline():
    if not AV2A_METRICS_PATH.exists():
        return {}
    data = json.load(open(AV2A_METRICS_PATH))
    overall = data.get("overall", data)
    return {
        "f1_audio_stage1": as_fraction(overall["F_seg_a"]),
        "f1_audio_stage2": as_fraction(overall["F_seg_a"]),
        "f1_visual_stage1": as_fraction(overall["F_seg_v"]),
        "f1_visual_stage2": as_fraction(overall["F_seg_v"]),
        "label": AV2A_METRICS_PATH.parent.name,
    }


AV2A_BASELINE = load_av2a_baseline()


fig, axes = plt.subplots(2, 3, figsize=(16, 8.5), facecolor="white")

panels = [
    ("audio_stage1", "f1_audio_stage1", "Audio stage1 F1",   "F1"),
    ("audio_stage2", "f1_audio_stage2", "Audio stage2 F1 (cross-modal hint)", "F1"),
    ("visual_stage1", "f1_visual_stage1", "Visual stage1 F1", "F1"),
    ("visual_stage2", "f1_visual_stage2", "Visual stage2 F1 (cross-modal hint)", "F1"),
    ("recon_audio_stage1", "recon_audio_stage1", "Audio recon (Step-4)",   "cos(ẑ, z)"),
    ("recon_visual_stage1", "recon_visual_stage1", "Visual recon (Step-4)", "cos(ẑ, z)"),
]

for ax, (_, key, title, ylabel) in zip(axes.flat, panels):
    for (name, rs, color, marker) in sources_ready:
        ax.plot(lams, col(rs, key), marker, color=color, label=name, linewidth=2.0, markersize=6)
    if key in AV2A_BASELINE:
        ax.axhline(
            AV2A_BASELINE[key],
            color="#333333",
            linestyle=":",
            linewidth=1.7,
            alpha=0.8,
            label=f"AV2A {AV2A_BASELINE['label']}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("λ_base")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)

fig.suptitle(f"Mean source comparison ({len(sources_ready)} sources) — visual mean = MSCOCO in all",
             fontsize=12, y=0.99)

out_png = OUT_DIR / "all_means.png"
out_pdf = OUT_DIR / "all_means.pdf"
fig.savefig(out_png, dpi=180, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"saved → {out_png}  ({len(sources_ready)} sources)")
print(f"saved → {out_pdf}")

# Summary table — wide form across all sources
summary = []
for i, lam in enumerate(lams):
    row = {"lambda": lam}
    for (name, rs, _, _) in sources_ready:
        tag = name.split("(")[0].strip().replace(" ", "_").replace("/", "_")
        row[f"{tag}__f1_a1"] = rs[i]["f1_audio_stage1"]
        row[f"{tag}__f1_a2"] = rs[i]["f1_audio_stage2"]
        row[f"{tag}__f1_v1"] = rs[i]["f1_visual_stage1"]
        row[f"{tag}__f1_v2"] = rs[i]["f1_visual_stage2"]
        row[f"{tag}__recon_a"] = rs[i]["recon_audio_stage1"]
        row[f"{tag}__recon_v"] = rs[i]["recon_visual_stage1"]
    summary.append(row)
(OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"saved → {OUT_DIR / 'summary.json'}")
