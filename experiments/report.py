"""Aggregate continual-learning results into a markdown table + plots."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from statistics import mean, pstdev

from common import BASE_SKILLS, NEW_SKILLS, RUNS

ORDER = ["finetune", "finetune_replay", "lora_merge", "rx_merge_only", "rx_gpm", "rx_unprojected", "rx", "rx_audit",
         "rx_grow"]


def metrics(run):
    M = run["matrix"]
    final = M[-1]
    base_k = run.get("base_skills", BASE_SKILLS)
    new_k = run.get("new_skills", NEW_SKILLS)
    allk = base_k + new_k
    learned_acc = [M[i][s] for i, s in enumerate(new_k)]  # right after learning
    new_final = [final[s] for s in new_k]
    base_final = [final[s] for s in base_k]
    base_init = [run["initial"][s] for s in base_k]
    # backward transfer on new skills (final - just learned), and on base skills
    bwt_new = mean(new_final[i] - learned_acc[i] for i in range(len(new_k) - 1))
    return {
        "avg_all": mean(final[k] for k in allk),
        "avg_base": mean(base_final),
        "avg_new": mean(new_final),
        "learn_acc": mean(learned_acc),
        "base_forgetting": mean(b0 - b for b0, b in zip(base_init, base_final)),
        "bwt_new": bwt_new,
        "minutes": run["seconds"] / 60,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(RUNS, "continual"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    runs = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(args.dir, "*.json"))):
        r = json.load(open(f))
        runs[r["method"]].append(r)
    names = [m for m in ORDER if m in runs] + [m for m in runs if m not in ORDER]
    cols = ["avg_all", "avg_base", "avg_new", "learn_acc", "base_forgetting", "bwt_new", "minutes"]
    lines = ["| method | seeds | " + " | ".join(cols) + " |",
             "|---|---|" + "---|" * len(cols)]
    for n in names:
        ms = [metrics(r) for r in runs[n]]
        cells = []
        for c in cols:
            vals = [m[c] for m in ms]
            s = f"{mean(vals):.3f}" if c != "minutes" else f"{mean(vals):.1f}"
            if len(vals) > 1 and c != "minutes":
                s += f" ± {pstdev(vals):.3f}"
            cells.append(s)
        lines.append(f"| {n} | {len(ms)} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)
    print(table)
    with open(os.path.join(args.dir, "summary.md"), "w") as f:
        f.write(table + "\n")
    if args.plot:
        plot(runs, names, args.dir)


def plot(runs, names, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    r0 = runs[names[0]][0]
    new_k = r0.get("new_skills", NEW_SKILLS)
    base_k = r0.get("base_skills", BASE_SKILLS)
    steps = list(range(1, len(new_k) + 1))
    for n in names:
        r = runs[n][0]
        base_curve = [mean(row[s] for s in base_k) for row in r["matrix"]]
        seen_curve = [mean(row[s] for s in new_k[: i + 1]) for i, row in enumerate(r["matrix"])]
        axes[0].plot(steps, base_curve, marker="o", label=n)
        axes[1].plot(steps, seen_curve, marker="o", label=n)
    axes[0].set_title("base skills (pre-trained) – mean accuracy")
    axes[1].set_title("new skills learned so far – mean accuracy")
    for ax in axes:
        ax.set_xticks(steps, new_k, rotation=30)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "continual.png"), dpi=120)

    # accuracy matrices
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), squeeze=False)
    allk = base_k + new_k
    for ax, n in zip(axes[0], names):
        r = runs[n][0]
        mat = [[row[k] for k in allk] for row in r["matrix"]]
        ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_title(n, fontsize=9)
        ax.set_xticks(range(len(allk)), allk, rotation=90, fontsize=6)
        ax.set_yticks(range(len(new_k)), [f"after {s}" for s in new_k], fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "matrices.png"), dpi=120)


if __name__ == "__main__":
    main()
