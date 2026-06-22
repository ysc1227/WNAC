from pathlib import Path

import matplotlib.pyplot as plt


SEQS = {
    "original": [0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 1.00],
    "ws_trial_920": [0.30, 0.43, 0.57, 0.70, 0.82, 0.93, 1.00],
    "ws_930_jagged": [0.30, 0.43, 0.57, 0.75, 0.82, 0.93, 1.00],
    "ws_trial_920_2": [0.30, 0.42, 0.56, 0.70, 0.83, 0.94, 1.00],
    "final_ws_930": [0.30, 0.44, 0.58, 0.72, 0.84, 0.92, 1.00],
}


def wavescale_sum(values):
    return values[0] + 2 * sum(values[1:])


def diffs(values):
    return [values[i + 1] - values[i] for i in range(len(values) - 1)]


def plot_all(out_dir):
    x = range(7)
    plt.figure(figsize=(13, 8))
    for name, values in SEQS.items():
        highlight = name in {"original", "ws_trial_920", "ws_trial_920_2"}
        plt.plot(
            x,
            values,
            marker="o" if highlight else ".",
            linewidth=3.0 if highlight else 1.2,
            alpha=1.0 if highlight else 0.45,
            label=f"{name} | ws={wavescale_sum(values):.2f}",
        )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("Scale factor candidates")
    plt.xlabel("scale index")
    plt.ylabel("scale_factor")
    plt.xticks(list(x))
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    path = out_dir / "scale_factor_candidates_all.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_selected(out_dir):
    selected = ["original", "ws_930_jagged", "final_ws_930"]
    x = range(7)
    plt.figure(figsize=(9, 6))
    for name in selected:
        values = SEQS[name]
        plt.plot(
            x,
            values,
            marker="o",
            linewidth=2.5,
            label=f"{name} | ws={wavescale_sum(values):.2f}",
        )
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.title("Weighted-sum matching candidates")
    plt.xlabel("scale index")
    plt.ylabel("scale_factor")
    plt.xticks(list(x))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "scale_factor_candidates_ws_match.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def plot_differences(out_dir):
    selected = ["original", "ws_930_jagged", "final_ws_930"]
    x = range(6)
    plt.figure(figsize=(9, 5))
    for name in selected:
        plt.plot(x, diffs(SEQS[name]), marker="o", linewidth=2.5, label=name)
    plt.title("Adjacent differences")
    plt.xlabel("difference index i -> i+1")
    plt.ylabel("delta")
    plt.xticks(list(x))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path = out_dir / "scale_factor_differences_ws_match.png"
    plt.savefig(path, dpi=180)
    plt.close()
    return path


def write_summary(out_dir):
    path = out_dir / "scale_factor_candidates_summary.txt"
    with path.open("w") as f:
        for name, values in SEQS.items():
            f.write(
                f"{name:16s} ws={wavescale_sum(values):.2f} "
                f"min={min(values):.2f} max={max(values):.2f} "
                f"diffs={[round(d, 3) for d in diffs(values)]}\n"
            )
    return path


def main():
    out_dir = Path("results/scale_factor_plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [plot_all(out_dir), plot_selected(out_dir), plot_differences(out_dir), write_summary(out_dir)]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()