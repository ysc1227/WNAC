import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "dac": "#6b7280",
    "wnac": "#1f77b4",
    "upscale": "#ff7f0e",
    "snac": "#2ca02c",
}
LOG_FLOOR = 1e-3
MODEL_ORDER = ["dac", "snac_reproduced", "wnac21_p005", "upscale21_p005"]
MODEL_LABELS = {
    "dac": "DAC 8 kbps",
    "snac_reproduced": "SNAC repro.",
    "wnac21_p005": "WNAC 8 kbps",
    "upscale21_p005": "Upscale 8 kbps",
}
MODEL_COLORS = {
    "dac": COLORS["dac"],
    "snac_reproduced": COLORS["snac"],
    "wnac21_p005": COLORS["wnac"],
    "upscale21_p005": COLORS["upscale"],
}
MODEL_MARKERS = {
    "dac": "s",
    "snac_reproduced": "^",
    "wnac21_p005": "o",
    "upscale21_p005": "o",
}


def load_blocks(path: Path, model: str):
    data = json.loads(path.read_text())
    return data["models"][model]["blocks"]


def values(rows, key):
    return [math.nan if row[key] is None else row[key] for row in rows]


def log_values(rows, key):
    return [math.nan if row[key] is None else max(LOG_FLOOR, row[key]) for row in rows]


def labels(rows):
    return [row["label"] for row in rows]


def plot_line(ax, rows, y, color, label, marker):
    x = list(range(len(rows)))
    ax.plot(x, y, marker=marker, lw=2.5, ms=6.5, color=color, label=label)


def format_block_axis(ax, rows):
    x = list(range(len(rows)))
    ax.set_xticks(x)
    ax.set_xticklabels(labels(rows))
    ax.set_xlim(-0.15, len(rows) - 0.85)
    if len(rows) > 6:
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--block-json",
        type=Path,
        default=Path("runs/analysis/dac_wnac21_p005_upscale21_p005_stage_block_probe_best_n128.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/analysis/stage_block_importance_leave_one_out.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("runs/analysis/stage_block_importance_leave_one_out.pdf"),
    )
    args = parser.parse_args()

    raw_data = json.loads(args.block_json.read_text())
    models = []
    for name in MODEL_ORDER:
        if name in raw_data["models"]:
            models.append((name, raw_data["models"][name]["blocks"]))
    for name, data in raw_data["models"].items():
        if name not in {model_name for model_name, _ in models}:
            models.append((name, data["blocks"]))

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.82, wspace=0.22)

    ax = axes[0]
    for name, rows in models:
        plot_line(
            ax,
            rows,
            values(rows, "block_audio_rms"),
            MODEL_COLORS.get(name, "#111827"),
            MODEL_LABELS.get(name, name),
            MODEL_MARKERS.get(name, "o"),
        )
    ax.set_title("Block Contribution")
    ax.set_xlabel("RVQ stage progress block")
    ax.set_ylabel("Waveform change RMS")
    format_block_axis(ax, models[0][1])
    ax.grid(True, alpha=0.28)

    ax = axes[1]
    for name, rows in models:
        plot_line(
            ax,
            rows,
            log_values(rows, "omit_extra_total_norm"),
            MODEL_COLORS.get(name, "#111827"),
            MODEL_LABELS.get(name, name),
            MODEL_MARKERS.get(name, "o"),
        )
    ax.set_yscale("log")
    ax.set_title("Block Leave-One-Out")
    ax.set_xlabel("RVQ stage progress block")
    ax.set_ylabel("Error increase")
    ax.set_ylim(LOG_FLOOR, None)
    format_block_axis(ax, models[0][1])
    ax.grid(True, which="both", alpha=0.28)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=max(1, len(labels_)),
        frameon=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
