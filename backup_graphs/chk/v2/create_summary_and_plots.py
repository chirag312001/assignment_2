#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt

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

                # parse filename: stats_<workload>_<predictor>.txt
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
    # --- Plot 1: IPC bar chart per workload ---
    for wl in df["workload"].unique():
        sub = df[df["workload"] == wl]
        plt.bar(sub["predictor"], sub["ipc"])
        plt.title(f"IPC for {wl}")
        plt.ylabel("IPC")

        # --- Zoom y-axis to highlight differences ---
        ymin = sub["ipc"].min() * 0.98   # 2% below smallest
        ymax = sub["ipc"].max() * 1.02   # 2% above largest
        plt.ylim(ymin, ymax)

        # --- Add numeric labels on bars ---
        for i, v in enumerate(sub["ipc"]):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        plt.savefig(os.path.join(out_dir, f"ipc_{wl}.png"))
        plt.clf()

    # --- Plot 2: Scatter mispred rate vs IPC ---
        # --- Plot 2: Scatter mispred rate vs IPC ---
    plt.figure(figsize=(6, 5))
    workloads = df["workload"].unique()
    colors = ["red", "blue", "green", "orange", "purple"]

    for i, wl in enumerate(workloads):
        sub = df[df["workload"] == wl]
        plt.scatter(sub["mispred_rate"], sub["ipc"],
                    s=80,  # point size
                    c=colors[i % len(colors)],
                    label=wl,
                    alpha=0.7)
        # add labels for each predictor
        for _, r in sub.iterrows():
            plt.annotate(r["predictor"],
                         (r["mispred_rate"], r["ipc"]),
                         fontsize=8, alpha=0.8)

    plt.xlabel("Branch misprediction rate")
    plt.ylabel("IPC")
    plt.title("IPC vs Branch misprediction rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ipc_vs_mispred.png"))
    plt.clf()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    main(args.results_dir, args.out_dir)
