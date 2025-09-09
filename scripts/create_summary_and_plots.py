#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 12})  # bigger font for reports

def try_parse_num(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return None

def parse_stats_file(path):
    stats = {}
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, val = parts[0], parts[1]
            if "simSeconds" in name:
                stats["sim_seconds"] = try_parse_num(val)
            elif "simInsts" in name:
                stats["committed_insts"] = try_parse_num(val)
            elif ".ipc" in name or name.endswith("IPC"):
                stats["ipc"] = try_parse_num(val)
            elif "branchPred.lookups" in name:
                stats["branch_predicted"] = try_parse_num(val)
            elif "commit.branchMispredicts" in name:
                stats["branch_mispredicted"] = try_parse_num(val)
    return stats

def main(results_dir, out_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.startswith("stats_") and f.endswith(".txt"):
                path = os.path.join(root, f)
                stats = parse_stats_file(path)

                name = f[len("stats_"):-len(".txt")]
                parts = name.split("_")
                workload = parts[0] if len(parts) > 0 else "unknown"
                predictor = parts[1] if len(parts) > 1 else "default"

                if stats.get("branch_predicted", 0):
                    mispred_rate = stats.get("branch_mispredicted", 0) / stats["branch_predicted"]
                else:
                    mispred_rate = None

                rows.append({
                    "workload": workload,
                    "predictor": predictor,
                    **stats,
                    "mispred_rate": mispred_rate
                })

    if not rows:
        print("No stats files found in", results_dir)
        return

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary.csv")
    df.to_csv(out_csv, index=False)
    print("✅ Saved", out_csv, "with", len(df), "rows")
    print(df)

    # --- Plot 1: IPC bar chart per workload ---
    for wl in df["workload"].unique():
        sub = df[df["workload"] == wl].sort_values("ipc", ascending=False)
        colors = plt.cm.Set2(range(len(sub)))

        plt.bar(sub["predictor"], sub["ipc"], color=colors)
        plt.title(f"IPC for {wl}")
        plt.ylabel("IPC")

        ymin = sub["ipc"].min() * 0.98
        ymax = sub["ipc"].max() * 1.02
        plt.ylim(ymin, ymax)
        plt.grid(axis="y", linestyle="--", alpha=0.7)

        for i, v in enumerate(sub["ipc"]):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ipc_{wl}.png"))
        plt.clf()

    # --- Plot 2: Scatter mispred rate vs IPC ---
    plt.figure(figsize=(6, 5))
    workloads = df["workload"].unique()
    colors = ["red", "blue", "green", "orange", "purple"]
    markers = ["o", "s", "^", "D", "x"]

    for i, wl in enumerate(workloads):
        sub = df[df["workload"] == wl]
        for j, predictor in enumerate(sub["predictor"].unique()):
            sp = sub[sub["predictor"] == predictor]
            plt.scatter(sp["mispred_rate"], sp["ipc"],
                        s=120,
                        c=colors[i % len(colors)],
                        marker=markers[j % len(markers)],
                        label=f"{wl}_{predictor}",
                        alpha=0.7)
    plt.xlabel("Branch misprediction rate")
    plt.ylabel("IPC")
    plt.title("IPC vs Branch misprediction rate")
    plt.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ipc_vs_mispred.png"))
    plt.clf()

    # --- Plot 3: Grouped bar chart (workloads × predictors) ---
    predictors = df["predictor"].unique()
    workloads = df["workload"].unique()
    x = np.arange(len(predictors))
    width = 0.35

    for i, wl in enumerate(workloads):
        sub = df[df["workload"] == wl].set_index("predictor").reindex(predictors)
        plt.bar(x + i*width, sub["ipc"], width, label=wl)

    plt.xticks(x + width/2, predictors)
    plt.ylabel("IPC")
    plt.title("IPC comparison across predictors & workloads")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ipc_grouped.png"))
    plt.clf()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    main(args.results_dir, args.out_dir)
