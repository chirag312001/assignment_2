#!/usr/bin/env python3
"""
Analysis + plotting for gem5 stats with combined predictor-complexity plot.

Usage:
  source .venv/bin/activate
  python3 scripts/create_summary_and_plots.py --results-dir results --out-dir analysis
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

plt.rcParams.update({"font.size": 10, "figure.max_open_warning": 300})

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
LINESTYLES = ['-', '--', '-.', ':']
TAB10 = plt.colormaps.get_cmap("tab10")
SET2 = plt.colormaps.get_cmap("Set2")

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
    return stats

def parse_filename(fname):
    base = fname[len("stats_"):-len(".txt")]
    parts = base.split("_")
    if len(parts) >= 2 and parts[1] in ["small", "large"]:
        workload = parts[0] + "_" + parts[1]
        pred_idx = 2
    else:
        workload = parts[0]
        pred_idx = 1
    predictor = parts[pred_idx] if len(parts) > pred_idx else "default"
    rob = None; iq = None
    for p in parts:
        if p.startswith("ROB"): rob = p.replace("ROB", "")
        if p.startswith("numIQEntries"): iq = p.replace("numIQEntries", "")
    try:
        rob_int = int(rob) if rob not in (None,"") else None
    except:
        rob_int = None
    try:
        iq_int = int(iq) if iq not in (None,"") else None
    except:
        iq_int = None
    return workload, predictor, rob_int, iq_int

def safe_name(s):
    return str(s).replace(" ", "_").replace("/", "_")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def short_label(pred):
    mapping = {
        "bimode": "bim",
        "gshare": "gsh",
        "local": "loc",
        "tournament": "tourn",
        "default": "def"
    }
    return mapping.get(pred, pred)

def auto_ylim(values, lower_pad_factor=0.02, upper_pad_factor=0.01, min_span=0.001):
    vals = np.array([v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))])
    if vals.size == 0:
        return (0, 1)
    vmin = vals.min()
    vmax = vals.max()
    span = max(vmax - vmin, min_span)
    ymin = vmin - max(lower_pad_factor * span, 0.0005)
    ymax = vmax + max(upper_pad_factor * span, 0.0005)
    if vmin >= 0 and ymin < 0:
        ymin = max(0.0, vmin - 0.0005)
    return (ymin, ymax)

def main(results_dir, out_dir):
    rows = []
    for root, _, files in os.walk(results_dir):
        for f in sorted(files):
            if f.startswith("stats_") and f.endswith(".txt"):
                path = os.path.join(root, f)
                stats = parse_stats(path)
                workload, predictor, rob, iq = parse_filename(f)
                bp = stats.get("branch_predicted", 0)
                bm = stats.get("branch_mispredicted", 0)
                mispred_rate = (bm / bp) if bp else None
                rows.append({
                    "run_path": path,
                    "workload": workload,
                    "predictor": predictor,
                    "ROB": rob,
                    "IQEntries": iq,
                    **stats,
                    "mispred_rate": mispred_rate
                })

    if not rows:
        print("No stats files found under", results_dir)
        return

    df = pd.DataFrame(rows)
    ensure_dir(out_dir)
    summary_csv = os.path.join(out_dir, "full_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"✅ Wrote {summary_csv} ({len(df)} rows)")

    # Per-workload IPC bar chart
    for wl in df["workload"].unique():
        sub = df[df["workload"] == wl].sort_values("ipc", ascending=False)
        if sub["ipc"].isnull().all(): continue
        fig, ax = plt.subplots(figsize=(14,6))
        colors = [SET2(i % 8) for i in range(len(sub))]
        x = np.arange(len(sub))
        bars = ax.bar(x, sub["ipc"], color=colors)
        ax.set_title(f"IPC for {wl}")
        ax.set_ylabel("IPC")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(p) for p in sub["predictor"].astype(str)], rotation=35, ha="right")
        ymin, ymax = auto_ylim(sub["ipc"].values)
        ax.set_ylim(ymin, ymax)
        for i, rect in enumerate(bars):
            h = rect.get_height()
            if not np.isnan(h):
                rob_val = sub.iloc[i]["ROB"]
                top_label = f"ROB{rob_val}\n{h:.3f}"
                ax.text(rect.get_x() + rect.get_width()/2.0, h, top_label, ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        plt.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.24)
        outp = os.path.join(out_dir, f"ipc_{safe_name(wl)}.png")
        fig.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close(fig)

    # Scatter: IPC vs mispred
    workloads = list(df["workload"].unique())
    fig, ax = plt.subplots(figsize=(12,8))
    cmap = TAB10
    for i, wl in enumerate(workloads):
        sub = df[df["workload"] == wl]
        if sub.empty: continue
        color = cmap(i % cmap.N)
        ax.scatter(sub["mispred_rate"], sub["ipc"], s=140, color=color, alpha=0.85, label=wl, edgecolor='k', linewidth=0.3)
        for _, r in sub.iterrows():
            if pd.notnull(r["mispred_rate"]) and pd.notnull(r["ipc"]):
                ax.annotate(short_label(r["predictor"]), (r["mispred_rate"], r["ipc"]), fontsize=8, alpha=0.9)
    ax.set_xlabel("Branch misprediction rate")
    ax.set_ylabel("IPC")
    ax.set_title("IPC vs Branch misprediction rate")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.subplots_adjust(left=0.06, right=0.78, top=0.95, bottom=0.06)
    fig.savefig(os.path.join(out_dir, "ipc_vs_mispred.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Grouped IPC by ROB per workload & IQ (annotated)
    for wl in df["workload"].unique():
        df_w = df[df["workload"] == wl]
        iq_values = sorted(df_w["IQEntries"].dropna().unique())
        rob_values = sorted(df_w["ROB"].dropna().unique())
        predictors = list(df_w["predictor"].unique())
        if not rob_values or not iq_values: continue
        for iq in iq_values:
            sub_iq = df_w[df_w["IQEntries"] == iq]
            if sub_iq.empty: continue
            pivot = sub_iq.pivot_table(index="predictor", columns="ROB", values="ipc")
            pivot = pivot.reindex(predictors).fillna(np.nan)
            x = np.arange(len(pivot.index))
            width = 0.8 / max(1, len(pivot.columns))
            fig, ax = plt.subplots(figsize=(14,6))
            for j, rob in enumerate(pivot.columns):
                vals = pivot[rob].values
                rects = ax.bar(x + j*width, vals, width=width, label=f"ROB{rob}", color=SET2(j % 8))
                for k, rect in enumerate(rects):
                    h = rect.get_height()
                    if not np.isnan(h):
                        label = f"ROB{rob}\n{h:.3f}"
                        ax.text(rect.get_x() + rect.get_width()/2.0, h, label, ha="center", va="bottom", fontsize=8)
            ax.set_xticks(x + width*(len(pivot.columns)-1)/2)
            ax.set_xticklabels([short_label(p) for p in pivot.index], rotation=35, ha="right")
            all_vals = pivot.values.flatten()
            ymin, ymax = auto_ylim(all_vals)
            ax.set_ylim(ymin, ymax)
            ax.set_ylabel("IPC")
            ax.set_title(f"IPC for {wl} (IQ={iq}) — grouped by ROB (labels show ROB and IPC)")
            ax.legend(title="ROB", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=9)
            ax.grid(axis="y", linestyle="--", alpha=0.6)
            plt.subplots_adjust(left=0.06, right=0.78, top=0.92, bottom=0.24)
            outp = os.path.join(out_dir, f"ipc_{safe_name(wl)}_IQ{iq}_byROB.png")
            fig.savefig(outp, dpi=160, bbox_inches="tight")
            plt.close(fig)

    # Clubbed scaling charts per workload (all predictors on one axes)
    for wl in df["workload"].unique():
        df_w = df[df["workload"] == wl]
        preds = sorted(df_w["predictor"].unique())
        # vs ROB
        if df_w["ROB"].notnull().any():
            fig, ax = plt.subplots(figsize=(12,6))
            plotted = False
            for idx, pred in enumerate(preds):
                dsub = df_w[(df_w["predictor"] == pred)].dropna(subset=["ROB","mispred_rate"]).sort_values("ROB")
                if dsub.empty: continue
                color = TAB10(idx % TAB10.N)
                marker = MARKERS[idx % len(MARKERS)]
                ls = LINESTYLES[idx % len(LINESTYLES)]
                ax.plot(dsub["ROB"], dsub["mispred_rate"], marker=marker, linestyle=ls, color=color, label=short_label(pred), linewidth=1.6)
                plotted = True
            if plotted:
                ax.set_xlabel("ROB size")
                ax.set_ylabel("Misprediction rate")
                ax.set_title(f"Mispred rate vs ROB — {wl}")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left")
                plt.subplots_adjust(left=0.08, right=0.78, top=0.94, bottom=0.12)
                outp = os.path.join(out_dir, f"mispred_vs_ROB_{safe_name(wl)}.png")
                fig.savefig(outp, dpi=160, bbox_inches="tight")
            plt.close(fig)
        # vs IQ
        if df_w["IQEntries"].notnull().any():
            fig, ax = plt.subplots(figsize=(12,6))
            plotted = False
            for idx, pred in enumerate(preds):
                dsub = df_w[(df_w["predictor"] == pred)].dropna(subset=["IQEntries","mispred_rate"]).sort_values("IQEntries")
                if dsub.empty: continue
                color = TAB10(idx % TAB10.N)
                marker = MARKERS[idx % len(MARKERS)]
                ls = LINESTYLES[idx % len(LINESTYLES)]
                ax.plot(dsub["IQEntries"], dsub["mispred_rate"], marker=marker, linestyle=ls, color=color, label=short_label(pred), linewidth=1.6)
                plotted = True
            if plotted:
                ax.set_xlabel("IQEntries")
                ax.set_ylabel("Misprediction rate")
                ax.set_title(f"Mispred rate vs IQ — {wl}")
                ax.grid(True, linestyle="--", alpha=0.5)
                ax.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left")
                plt.subplots_adjust(left=0.08, right=0.78, top=0.94, bottom=0.12)
                outp = os.path.join(out_dir, f"mispred_vs_IQ_{safe_name(wl)}.png")
                fig.savefig(outp, dpi=160, bbox_inches="tight")
            plt.close(fig)

    # -------------- NEW: Combined plot - misprediction rate vs predictor complexity (all workloads) --------------
    complexity_order = ["bimode", "gshare", "local", "tournament"]
    # only include predictors that appear in data
    available_preds = [p for p in complexity_order if p in df["predictor"].unique()]
    if len(available_preds) >= 2:
        fig, ax = plt.subplots(figsize=(14,8))
        workloads = sorted(df["workload"].unique())
        for w_idx, wl in enumerate(workloads):
            sub = df[df["workload"] == wl]
            # compute mean mispred_rate per predictor type (in case of multiple ROB/IQ combos)
            y_vals = []
            for pred in available_preds:
                vals = sub[sub["predictor"] == pred]["mispred_rate"].dropna().values
                if vals.size == 0:
                    y_vals.append(np.nan)
                else:
                    y_vals.append(np.mean(vals))
            # plot line for this workload
            color = TAB10(w_idx % TAB10.N)
            marker = MARKERS[w_idx % len(MARKERS)]
            ls = LINESTYLES[w_idx % len(LINESTYLES)]
            # skip if all NaN
            if np.all(np.isnan(y_vals)):
                continue
            ax.plot(available_preds, y_vals, marker=marker, linestyle=ls, color=color, label=wl, linewidth=1.6)
        ax.set_xlabel("Predictor (complexity order → more complex)")
        ax.set_ylabel("Average misprediction rate")
        ax.set_title("Misprediction rate vs Predictor Complexity (all workloads)")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=9, bbox_to_anchor=(1.02,1), loc="upper left")
        plt.subplots_adjust(left=0.08, right=0.78, top=0.94, bottom=0.12)
        outp = os.path.join(out_dir, "mispred_vs_complexity_all_workloads.png")
        fig.savefig(outp, dpi=160, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ All plots saved in {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results")
    p.add_argument("--out-dir", default="analysis")
    args = p.parse_args()
    main(args.results_dir, args.out_dir)

