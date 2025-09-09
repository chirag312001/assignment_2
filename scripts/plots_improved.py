#!/usr/bin/env python3
# scripts/plots_improved.py
import pandas as pd, numpy as np, os, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

IN = "analysis/detailed_summary_table.csv"
OUTDIR = "analysis/plots_improved"
os.makedirs(OUTDIR, exist_ok=True)

df = pd.read_csv(IN)

# ---------- define predictor complexity ----------
# You can tweak these weights:
predictor_weights = {"local": 1.0, "gshare": 1.5, "bimode": 2.0, "tournament": 2.5, "default":1.0}
def complexity_score(row):
    base = predictor_weights.get(str(row['predictor']).lower(), 1.0)
    rob = row.get("ROB_config") if not pd.isna(row.get("ROB_config")) else 0
    iq = row.get("IQ_config") if not pd.isna(row.get("IQ_config")) else 0
    # scale IQ smaller so complexity doesn't blow up: iq/100
    return base + (rob or 0)/100.0 + (iq or 0)/1000.0

df['complexity'] = df.apply(complexity_score, axis=1)

# helper to auto-zoom a little around values if values are close
def auto_ylim(vals, pad_ratio=0.06, min_span=0.01):
    vals = np.array(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals)==0:
        return (0,1)
    mn, mx = vals.min(), vals.max()
    span = max(mx-mn, min_span)
    pad = span * pad_ratio
    return (max(0.0, mn - pad), mx + pad)

# ---------- 1) Bar chart: IPC grouped by predictor/ROB for each workload (grouping by IQ) ----------
for (wl, iq), sub in df.groupby(['workload','IQ_config']):
    sub = sub.sort_values('ipc', ascending=False)
    if sub['ipc'].isnull().all(): continue
    # group by ROB for same predictor
    pivot = sub.pivot_table(index='predictor', columns='ROB_config', values='ipc')
    pivot = pivot.fillna(np.nan)
    n_preds = len(pivot.index)
    n_robs = len(pivot.columns)
    fig, ax = plt.subplots(figsize=(max(8, n_preds*0.6 + 2), 6))
    x = np.arange(n_preds)
    width = 0.8 / max(1, n_robs)
    colors = plt.get_cmap("tab10")
    for j, rob in enumerate(pivot.columns):
        vals = pivot[rob].values
        rects = ax.bar(x + j*width, vals, width=width, label=f"ROB{rob}", color=colors(j % 10))
        # annotate each bar
        for rect, val in zip(rects, vals):
            if np.isnan(val): continue
            ax.text(rect.get_x() + rect.get_width()/2, val + 0.002*max(1, np.nanmax(vals)), f"{val:.3f}\nIQ{iq}", ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width*(n_robs-1)/2)
    ax.set_xticklabels([str(p) for p in pivot.index], rotation=30, ha='right')
    ax.set_ylabel("IPC")
    ax.set_title(f"IPC for {wl} (IQ={iq}) - grouped by ROB")
    ymin,ymax = auto_ylim(pivot.values.flatten())
    ax.set_ylim(ymin,ymax)
    ax.legend(title="ROB", bbox_to_anchor=(1.02,1), loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR,f"ipc_{wl}_IQ{iq}_byROB.png"), dpi=180, bbox_inches='tight')
    plt.close(fig)

# ---------- 2) Combined per-workload IPC (all predictors in one line plot; helpful when many points) ----------
for wl, sub in df.groupby('workload'):
    sub = sub.dropna(subset=['ipc'])
    if sub.empty: continue
    fig, ax = plt.subplots(figsize=(10,6))
    # create a label that includes predictor + ROB + IQ
    sub['label'] = sub.apply(lambda r: f"{r['predictor']}_ROB{int(r['ROB_config']) if not pd.isna(r['ROB_config']) else 'NA'}_IQ{int(r['IQ_config']) if not pd.isna(r['IQ_config']) else 'NA'}", axis=1)
    sub = sub.sort_values('ipc', ascending=False)
    ax.plot(range(len(sub)), sub['ipc'], marker='o', linestyle='-', linewidth=1)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub['label'], rotation=60, ha='right', fontsize=7)
    ax.set_ylabel("IPC")
    ax.set_title(f"IPC across predictors for {wl}")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"ipc_all_predictors_{wl}.png"), dpi=160, bbox_inches='tight')
    plt.close(fig)

# ---------- 3) Line chart: misprediction rate vs predictor complexity ----------
# we will compute mean mispred_rate per predictor+config combo (group by predictor and ROB/IQ)
group = df.dropna(subset=['mispred_rate']).copy()
if not group.empty:
    # compute avg mispred_rate per predictor config (ROB+IQ)
    # for x-axis use complexity score; for readability, we'll average mispred_rate for identical complexity
    group['config_id'] = group.apply(lambda r: f"{r['predictor']}_ROB{int(r['ROB_config']) if not pd.isna(r['ROB_config']) else 'NA'}_IQ{int(r['IQ_config']) if not pd.isna(r['IQ_config']) else 'NA'}", axis=1)
    agg = group.groupby(['config_id']).agg({
        'complexity': 'mean',
        'mispred_rate': 'mean',
        'ipc': 'mean',
        'predictor': lambda s: s.iloc[0]
    }).reset_index().sort_values('complexity')
    fig, ax = plt.subplots(figsize=(8,5))
    # different line styles per predictor type
    styles = {'local':'-','gshare':'--','bimode':':','tournament':'-.'}
    markers = {'local':'o','gshare':'s','bimode':'^','tournament':'x'}
    preds = agg['predictor'].unique()
    for p in preds:
        sub2 = agg[agg['predictor']==p]
        ax.plot(sub2['complexity'], sub2['mispred_rate'], label=p, marker=markers.get(p,'o'), linestyle=styles.get(p,'-'))
    ax.set_xlabel("Complexity (score)")
    ax.set_ylabel("Misprediction rate")
    ax.set_title("Mispred rate vs predictor complexity (all configs)")
    ax.legend(title="Predictor")
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR,"mispred_vs_complexity.png"), dpi=160)
    plt.close(fig)

# ---------- 4) Scatter plot: misprediction rate vs IPC (colour by predictor type) ----------
if not group.empty:
    fig, ax = plt.subplots(figsize=(8,6))
    pred_types = group['predictor'].unique()
    cmap = plt.get_cmap("tab10")
    for i,p in enumerate(pred_types):
        s2 = group[group['predictor']==p]
        ax.scatter(s2['mispred_rate'], s2['ipc'], label=p, alpha=0.8, s=40, marker='o')
    ax.set_xlabel("Misprediction rate")
    ax.set_ylabel("IPC")
    ax.set_title("Mispred rate vs IPC (each point is a config)")
    ax.legend(title="Predictor")
    ax.grid(linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR,"scatter_mispred_vs_ipc.png"), dpi=160)
    plt.close(fig)

print("Plots written to", OUTDIR)
