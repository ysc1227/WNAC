import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


COLORS = {
    "wnac": "#1f77b4",
    "upscale": "#ff7f0e",
    "dac": "#6b7280",
}
LOG_FLOOR = 1e-3


def load_stages(path: Path, model: str):
    data = json.loads(path.read_text())
    return data["models"][model]["stages"]


def values(stages, key):
    return [row[key] for row in stages]


def log_values(stages, key):
    return [max(LOG_FLOOR, row[key]) for row in stages]


def stage_progress(stages):
    if len(stages) <= 1:
        return [0.0 for _ in stages]
    denom = len(stages) - 1
    return [i / denom for i, _ in enumerate(stages)]


def format_progress_axis(ax):
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wnac-upscale-json",
        type=Path,
        default=Path("runs/analysis/wnac21_p005_vs_upscale21_p005_stage_frequency_probe_best_n128.json"),
    )
    parser.add_argument(
        "--dac-json",
        type=Path,
        default=Path("runs/analysis/dac_9_1s_vs_wavescale_stage_frequency_probe_best_n128.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/analysis/stage0_importance_leave_one_out.png"),
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=Path("runs/analysis/stage0_importance_leave_one_out.pdf"),
    )
    args = parser.parse_args()

    wnac = load_stages(args.wnac_upscale_json, "wnac21_p005")
    upscale = load_stages(args.wnac_upscale_json, "upscale21_p005")
    dac = load_stages(args.dac_json, "dac")

    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.82, wspace=0.22)

    ax = axes[0]
    ax.plot(
        stage_progress(dac),
        values(dac, "marginal_audio_rms"),
        marker="s",
        lw=2.2,
        ms=5.0,
        color=COLORS["dac"],
        label="DAC 8 kbps",
    )
    ax.plot(
        stage_progress(wnac),
        values(wnac, "marginal_audio_rms"),
        marker="o",
        lw=2.4,
        ms=5.5,
        color=COLORS["wnac"],
        label="WNAC 8 kbps",
    )
    ax.plot(
        stage_progress(upscale),
        values(upscale, "marginal_audio_rms"),
        marker="o",
        lw=2.4,
        ms=5.5,
        color=COLORS["upscale"],
        label="Upscale 8 kbps",
    )
    ax.set_title("Contribution")
    ax.set_xlabel("RVQ stage progress")
    ax.set_ylabel("Marginal audio RMS")
    format_progress_axis(ax)
    ax.grid(True, alpha=0.28)

    ax = axes[1]
    ax.plot(
        stage_progress(dac),
        log_values(dac, "omit_extra_total_norm"),
        marker="s",
        lw=2.2,
        ms=5.0,
        color=COLORS["dac"],
        label="DAC 8 kbps",
    )
    ax.plot(
        stage_progress(wnac),
        log_values(wnac, "omit_extra_total_norm"),
        marker="o",
        lw=2.4,
        ms=5.5,
        color=COLORS["wnac"],
        label="WNAC 8 kbps",
    )
    ax.plot(
        stage_progress(upscale),
        log_values(upscale, "omit_extra_total_norm"),
        marker="o",
        lw=2.4,
        ms=5.5,
        color=COLORS["upscale"],
        label="Upscale 8 kbps",
    )
    ax.set_yscale("log")
    ax.set_title("Leave-One-Out")
    ax.set_xlabel("RVQ stage progress")
    ax.set_ylabel("Error increase")
    ax.set_ylim(LOG_FLOOR, None)
    format_progress_axis(ax)
    ax.grid(True, which="both", alpha=0.28)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=3, frameon=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    fig.savefig(args.pdf, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
