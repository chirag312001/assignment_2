#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 11})

def try_num(s):
    try:
        return int(s)
    except:
        try:
            return float(s)
        except:
            return None

def parse_stats(path):
    stats = {}
    with open(path, "r", errors="ignore") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            name, val = parts[0], parts[1]
            if "simSeconds" in name:
                stats["sim_seconds"] = try_num(val)
            elif "simInsts" in name:
                stats["committed_insts"] = try_num(val)
            elif ".ipc" in name or name.endswith("IPC"):
                stats["ipc"] = try_num(val)
            elif "branchPred.lookups" in name:
                stats["branch_predicted"] = try_num(val)
            elif "commit.branchMispredicts" in name:
                stats["branch_mispredicted"] = try_num(val)
            # add more keys here if you find recovery-cycle stats
    return stats

def parse_filename(fname):
    """Parse workload, predictor, ROB, IQEntries from stats filename"""
    name = fname[len("stats_"):-len(".txt")]
    parts = name.split("_")

    # workloads like quicksort_small or dijkstra_small
    if len(parts) >= 2 and parts[1] in ["small", "large"]:
        workload = parts[0] + "_" + parts[1]
        pred_idx = 2
    else:
        workload = parts[0]
        pred_idx = 1

    predictor = parts[pred_idx] if len(parts) > pred_idx else "default"

    rob = None
    iq = None
    for p in parts:
        if p.startswith("ROB"):
            rob = p.replace("ROB", "")
        if p.startswith("numIQEntries"):
            iq = p.replace("numIQEntries", "")

    # normalize types
    try:
        rob_int = int(rob) if rob is not None and rob != "" else None
    except:
        rob_int = None
    try:
        iq_int = int(iq) if iq is not None and iq != "" else None
    except:
        iq_int = None

    return workload, predictor, rob_int, iq_int

def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_")

def main(results_dir, out_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        for f in files:
            if f.startswith("stats_") and f.endswith(".txt"):
                path = os.path.join(root, f)
                stats = parse_stats(path)
                workload, predictor, rob, iq = parse_filename(f)

                bp = stats.get("branch_predicted", 0)
                bm = stats.get("branch_mispredicted", 0)
                mispred_rate = (bm / bp) if bp else None

                rows.append({
                    "workload": workload,
                    "predictor": predictor,
                    "ROB": rob,
                    "IQEntries": iq,
                    **stats,
                    "mispred_rate": mispred_rate
                })

    if not rows:
        print("No stats files found in", results_dir)
        return

    df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "full_summary.csv")
    df.to_csv(out_csv, index=False)
    print("✅ Wrote", out_csv, "with", len(df), "rows")
    print(df.head())

    # Basic IPC bar per workload (sorted & labeled)
    for wl in df["workload"].unique():
        sub = df[df["workload"] == wl].sort_values("ipc", ascending=False)
        plt.figure(figsize=(8, 4))
        colors = plt.cm.Set2(range(len(sub)))
        plt.bar(sub["predictor"].astype(str), sub["ipc"], color=colors)
        plt.title(f"IPC for {wl}")
        plt.ylabel("IPC")
        ymin = sub["ipc"].min() * 0.98
        ymax = sub["ipc"].max() * 1.02
        plt.ylim(ymin, ymax)
        for i, v in enumerate(sub["ipc"]):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(rotation=35, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"ipc_{safe_name(wl)}.png"))
        plt.clf()

    # Scatter: mispred rate vs IPC (color by workload, marker by predictor)
    plt.figure(figsize=(8, 6))
    workloads = list(df["workload"].unique())
    palette = plt.cm.tab10
    marker_list = ['o','s','^','D','v','P','X','*']
    predictor_order = list(df["predictor"].unique())
    for i, wl in enumerate(workloads):
        sub = df[df["workload"] == wl]
        plt.scatter(sub["mispred_rate"], sub["ipc"],
                    s=100,
                    c=[palette(i) for _ in range(len(sub))],
                    marker='o',
                    alpha=0.8,
                    label=wl)
        for _, r in sub.iterrows():
            plt.annotate(r["predictor"], (r["mispred_rate"], r["ipc"]), fontsize=8, alpha=0.9)
    plt.xlabel("Branch misprediction rate")
    plt.ylabel("IPC")
    plt.title("IPC vs Branch misprediction rate")
    plt.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ipc_vs_mispred.png"))
    plt.clf()

    # --- New: grouped bar charts by ROB for each workload and IQEntries ---
    # For each workload and each IQEntries value, produce a grouped bar chart:
    # x-axis = predictors, groups = ROB sizes
    for wl in df["workload"].unique():
        df_w = df[df["workload"] == wl]
        iq_values = sorted(df_w["IQEntries"].dropna().unique())
        rob_values = sorted(df_w["ROB"].dropna().unique())
        predictors = list(df_w["predictor"].unique())

        # If no ROB/IQ variation, skip grouped by ROB generation
        if len(rob_values) == 0 or len(iq_values) == 0:
            continue

        for iq in iq_values:
            sub_iq = df_w[df_w["IQEntries"] == iq]
            if sub_iq.empty:
                continue

            # Prepare data: rows = predictors, cols = ROB values
            pivot = sub_iq.pivot_table(index="predictor", columns="ROB", values="ipc")
            pivot = pivot.reindex(predictors).fillna(0)

            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, len(pivot.columns))
            plt.figure(figsize=(10, 4))
            for j, rob in enumerate(pivot.columns):
                vals = pivot[rob].values
                plt.bar(x + j*width, vals, width=width, label=f"ROB{rob}")
            plt.xticks(x + width*(len(pivot.columns)-1)/2, pivot.index, rotation=35, ha="right")
            plt.ylabel("IPC")
            plt.title(f"IPC for {wl} (IQ={iq}) — grouped by ROB")
            plt.legend(title="ROB")
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.tight_layout()
            fname = os.path.join(out_dir, f"ipc_{safe_name(wl)}_IQ{iq}_byROB.png")
            plt.savefig(fname)
            plt.clf()

    # --- New: scaling line charts (mispred_rate vs ROB and vs IQ) per workload & predictor ---
    for wl in df["workload"].unique():
        df_w = df[df["workload"] == wl]
        preds = df_w["predictor"].unique()
        for pred in preds:
            df_wp = df_w[df_w["predictor"] == pred]
            # vs ROB (for each IQ)
            if df_wp["ROB"].notnull().any():
                plt.figure(figsize=(6,4))
                for iq in sorted(df_wp["IQEntries"].dropna().unique()):
                    dsub = df_wp[df_wp["IQEntries"] == iq].dropna(subset=["ROB","mispred_rate"]).sort_values("ROB")
                    if dsub.empty:
                        continue
                    plt.plot(dsub["ROB"], dsub["mispred_rate"], marker='o', label=f"IQ={iq}")
                plt.xlabel("ROB size")
                plt.ylabel("Misprediction rate")
                plt.title(f"Mispred rate vs ROB — {wl} / {pred}")
                plt.grid(True); plt.legend(fontsize=8); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"mispred_vs_ROB_{safe_name(wl)}_{safe_name(pred)}.png"))
                plt.clf()

            # vs IQ (for each ROB)
            if df_wp["IQEntries"].notnull().any():
                plt.figure(figsize=(6,4))
                for rob in sorted(df_wp["ROB"].dropna().unique()):
                    dsub = df_wp[df_wp["ROB"] == rob].dropna(subset=["IQEntries","mispred_rate"]).sort_values("IQEntries")
                    if dsub.empty:
                        continue
                    plt.plot(dsub["IQEntries"], dsub["mispred_rate"], marker='o', label=f"ROB={rob}")
                plt.xlabel("IQEntries")
                plt.ylabel("Misprediction rate")
                plt.title(f"Mispred rate vs IQ — {wl} / {pred}")
                plt.grid(True); plt.legend(fontsize=8); plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"mispred_vs_IQ_{safe_name(wl)}_{safe_name(pred)}.png"))
                plt.clf()

    print("✅ All plots saved in", out_dir)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    main(args.results_dir, args.out_dir)
