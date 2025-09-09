#!/usr/bin/env python3
"""
generate_summary_table.py

Scan gem5 stats files (results/stats_*.txt) and produce:
 - analysis/detailed_summary_table.csv
 - analysis/detailed_summary_table.tex

This script is defensive: it searches for common substrings found in gem5 outputs
and records whichever values exist.

Usage:
    source .venv/bin/activate
    python3 scripts/generate_summary_table.py --results-dir results --out-dir analysis
"""
import os
import re
import argparse
import math
import pandas as pd
import numpy as np

# ---------------- Config ----------------
DEFAULT_RESULTS = "results"
DEFAULT_OUT = "analysis"
os.makedirs(DEFAULT_OUT, exist_ok=True)

# Map substrings (lowercased) -> canonical column name
# Add any more keys you see in your grep output
PATTERN_MAP = {
    # basic counters
    "simseconds": "sim_seconds",
    "siminsts": "committed_insts",
    # already used
    "ipc": "ipc",
    # branch counters
    "branchpred.lookups": "branch_predicted",
    "condpredicted": "branch_predicted",
    "commit.branchmispredicts": "branch_mispredicted",
    "branchmispredicts": "branch_mispredicted",

    # recovery cycles (if present)
    "branchrecoverycycles": "branch_recovery_cycles",
    "recoverycycles": "branch_recovery_cycles",
    "mispredict_recovery_cycles": "branch_recovery_cycles",
    "mispredictpenalty": "branch_recovery_cycles",
    "mispredict_penalty": "branch_recovery_cycles",

    # squash / flush / examined
    "squashedinstsissued": "squashedInstsIssued",
    "squashedinstsexamined": "squashedInstsExamined",
    "squashedoperandsexamined": "squashedOperandsExamined",
    "squash": "squash_generic",

    # stalls / bubbles / stall_time
    "fetchbubble": "fetch_bubbles",
    "fetch_bubbles": "fetch_bubbles",
    "m_stall_time": "avg_stall_time_field",  # average stall per message or total (we disambiguate below)
    ".m_stall_time": "m_stall_time_any",
    ".m_stall_count": "m_stall_count_any",
    "stall_time": "stall_time_generic",

    # ROB / IQ / issue queue proxies
    "instsadded": "instsAdded",               # number of insts added to IQ
    "numissueddist::mean": "numIssuedDist_mean",  # avg issued per cycle (proxy for issue rate)
    "iqfull": "IQ_full_count",
    "numiqentries": "IQ_entries_config",     # usually in filename but keep key

    # L1 caches (your files show these exact keys)
    "l1dcache.m_demand_accesses": "L1D_accesses",
    "l1dcache.m_demand_misses": "L1D_misses",
    "l1icache.m_demand_accesses": "L1I_accesses",
    "l1icache.m_demand_misses": "L1I_misses",
    "l1_controllers.l1dcache.m_demand_accesses": "L1D_accesses",
    "l1_controllers.l1dcache.m_demand_misses": "L1D_misses",
    "l1_controllers.l1icache.m_demand_accesses": "L1I_accesses",
    "l1_controllers.l1icache.m_demand_misses": "L1I_misses",

    # memory/dir stalls
    "requesttomemory.m_stall_time": "requestToMemory_m_stall_time",
    "mandatoryqueue.m_stall_time": "L1_mandatoryQueue_m_stall_time",
    "requesttomemory.m_avg_stall_time": "requestToMemory_m_avg_stall_time",
    "mandatoryqueue.m_avg_stall_time": "L1_mandatoryQueue_m_avg_stall_time",

    # other helpful network/memory stats (optional)
    "m_misslatencyhistseqr::mean": "mem_miss_latency_mean",
    "ifetch.miss_latency_hist_seqr::mean": "ifetch_miss_latency_mean",
}

# fallback substring matches (less specific) - these will be checked after pattern_map
FALLBACKS = [
    ("l1d", "L1D_accesses"),
    ("l1i", "L1I_accesses"),
    ("dcache", "L1D_accesses"),
    ("icache", "L1I_accesses"),
    ("squash", "squash_generic"),
    ("stall", "stall_time_generic"),
    ("fetch", "fetch_generic"),
    ("m_demand_misses", "L1D_misses"),
]

# helper to try convert to number
def try_num(s):
    try:
        if isinstance(s, (int, float)):
            return s
        return int(s)
    except Exception:
        try:
            return float(s)
        except Exception:
            return np.nan

def extract_number_from_line(line):
    # gem5 lines often look like:
    # stat.name    12345    # comment
    # or "name    1.234    # ..."
    # We'll look for the first numeric token.
    m = re.search(r"([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", line)
    if m:
        return try_num(m.group(1))
    return None

def parse_stats_file(path):
    """Return dict canonical_name -> value for stats found in file."""
    found = {}
    with open(path, "r", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("----"):
                continue
            low = line.lower()
            # check direct pattern map (exact substrings)
            matched = False
            for pat, key in PATTERN_MAP.items():
                if pat in low:
                    val = extract_number_from_line(line)
                    if val is not None:
                        # if key already present and is numeric, sum (useful for some counters)
                        prev = found.get(key)
                        if prev is None or (isinstance(prev, float) and np.isnan(prev)):
                            found[key] = val
                        else:
                            # sum counters by default
                            try:
                                found[key] = prev + val
                            except Exception:
                                found[key] = val
                    matched = True
                    break
            if matched:
                continue
            # fallbacks (less precise)
            for pat, key in FALLBACKS:
                if pat in low:
                    val = extract_number_from_line(line)
                    if val is not None:
                        prev = found.get(key)
                        if prev is None or (isinstance(prev, float) and np.isnan(prev)):
                            found[key] = val
                        else:
                            found[key] = prev + val
                    break
    return found

def parse_filename(fname):
    """Parse workload/predictor/ROB/IQ from filename like stats_quicksort_small_local_ROB128_numIQEntries64.txt"""
    base = fname
    if base.startswith("stats_"):
        base = base[6:]
    if base.endswith(".txt"):
        base = base[:-4]
    parts = base.split("_")
    # workload might be "quicksort_small" or "quicksort"
    workload = parts[0]
    if len(parts) >= 2 and parts[1] in ("small", "large"):
        workload = parts[0] + "_" + parts[1]
        pred_idx = 2
    else:
        pred_idx = 1
    predictor = parts[pred_idx] if len(parts) > pred_idx else "default"
    rob = None
    iq = None
    for p in parts:
        if p.startswith("ROB"):
            try:
                rob = int(p.replace("ROB", ""))
            except:
                rob = p.replace("ROB", "")
        if p.startswith("numIQEntries"):
            try:
                iq = int(p.replace("numIQEntries", ""))
            except:
                iq = p.replace("numIQEntries", "")
    return workload, predictor, rob, iq

def make_latex_table(df, out_path):
    # pick columns to show in report-friendly order
    cols = [
        "workload", "predictor", "ROB_config", "IQ_config",
        "ipc", "branch_accuracy", "mispred_rate",
        "avg_recovery_penalty_cycles",
        "avg_squashed_insts_per_mispred",
        "squashedInstsIssued", "squashedInstsExamined",
        "L1D_accesses", "L1D_misses", "L1I_accesses", "L1I_misses",
        "instsAdded", "numIssuedDist_mean",
        "requestToMemory_m_stall_time", "L1_mandatoryQueue_m_stall_time"
    ]
    # only include those present
    cols = [c for c in cols if c in df.columns]

    with open(out_path, "w") as tf:
        tf.write("\\begin{table}[ht]\n\\centering\n\\small\n")
        colspec = "l l c c " + " ".join(["r"]*(len(cols)-4))
        tf.write("\\begin{tabular}{" + colspec + "}\n")
        tf.write("\\hline\n")
        header = ["Workload","Predictor","ROB","IQ"] + [c.replace("_","\\_") for c in cols[4:]]
        tf.write(" & ".join(header) + " \\\\\n")
        tf.write("\\hline\n")
        for _, r in df.iterrows():
            cells = []
            for c in cols:
                v = r.get(c, "")
                if pd.isna(v):
                    cells.append("--")
                elif isinstance(v, float):
                    cells.append(f"{v:.4g}")
                else:
                    cells.append(str(v))
            tf.write(" & ".join(cells) + " \\\\\n")
        tf.write("\\hline\n")
        tf.write("\\end{tabular}\n")
        tf.write("\\caption{Detailed summary per workload/predictor (auto-generated).}\n")
        tf.write("\\label{tab:detailed_summary}\n")
        tf.write("\\end{table}\n")

def main(results_dir, out_dir):
    rows = []
    files_scanned = 0
    for root, _, files in os.walk(results_dir):
        for f in sorted(files):
            if not f.startswith("stats_") or not f.endswith(".txt"):
                continue
            files_scanned += 1
            path = os.path.join(root, f)
            meta = parse_filename(f)
            workload, predictor, rob, iq = meta
            parsed = parse_stats_file(path)

            sim_seconds = parsed.get("sim_seconds", np.nan)
            committed_insts = parsed.get("committed_insts", np.nan)
            ipc = parsed.get("ipc", np.nan)

            branch_predicted = parsed.get("branch_predicted", np.nan)
            branch_mispredicted = parsed.get("branch_mispredicted", np.nan)
            # recovery cycles if present
            recovery_cycles = parsed.get("branch_recovery_cycles", np.nan)

            # squashes & proxies
            squashedIssued = parsed.get("squashedInstsIssued", np.nan)
            squashedExamined = parsed.get("squashedInstsExamined", np.nan)
            squashedOperands = parsed.get("squashedOperandsExamined", np.nan)

            instsAdded = parsed.get("instsAdded", np.nan)
            issue_rate_mean = parsed.get("numIssuedDist_mean", np.nan)

            L1D_accesses = parsed.get("L1D_accesses", np.nan)
            L1D_misses = parsed.get("L1D_misses", np.nan)
            L1I_accesses = parsed.get("L1I_accesses", np.nan)
            L1I_misses = parsed.get("L1I_misses", np.nan)

            req_mem_stall = parsed.get("requestToMemory_m_stall_time", np.nan)
            req_mem_avg_stall = parsed.get("requestToMemory_m_avg_stall_time", np.nan)
            l1_mand_stall = parsed.get("L1_mandatoryQueue_m_stall_time", np.nan)
            l1_mand_avg_stall = parsed.get("L1_mandatoryQueue_m_avg_stall_time", np.nan)

            # derived metrics
            mispred_rate = np.nan
            if not (pd.isna(branch_predicted) or branch_predicted == 0):
                mispred_rate = (branch_mispredicted / branch_predicted) if not pd.isna(branch_mispredicted) else np.nan
            branch_accuracy = (1.0 - mispred_rate) if not pd.isna(mispred_rate) else np.nan

            avg_recovery_penalty_cycles = np.nan
            if not pd.isna(recovery_cycles) and not pd.isna(branch_mispredicted) and branch_mispredicted > 0:
                avg_recovery_penalty_cycles = recovery_cycles / branch_mispredicted
            # else leave NaN. We'll add a helpful proxy below.

            avg_squashed_insts_per_mispred = np.nan
            if not pd.isna(squashedExamined) and not pd.isna(branch_mispredicted) and branch_mispredicted > 0:
                avg_squashed_insts_per_mispred = squashedExamined / branch_mispredicted

            row = {
                "file": path,
                "workload": workload,
                "predictor": predictor,
                "ROB_config": rob,
                "IQ_config": iq,
                "sim_seconds": sim_seconds,
                "committed_insts": committed_insts,
                "ipc": ipc,
                "branch_predicted": branch_predicted,
                "branch_mispredicted": branch_mispredicted,
                "mispred_rate": mispred_rate,
                "branch_accuracy": branch_accuracy,
                "branch_recovery_cycles_total": recovery_cycles,
                "avg_recovery_penalty_cycles": avg_recovery_penalty_cycles,
                "squashedInstsIssued": squashedIssued,
                "squashedInstsExamined": squashedExamined,
                "squashedOperandsExamined": squashedOperands,
                "avg_squashed_insts_per_mispred": avg_squashed_insts_per_mispred,
                "instsAdded": instsAdded,
                "numIssuedDist_mean": issue_rate_mean,
                "L1D_accesses": L1D_accesses,
                "L1D_misses": L1D_misses,
                "L1I_accesses": L1I_accesses,
                "L1I_misses": L1I_misses,
                "requestToMemory_m_stall_time": req_mem_stall,
                "requestToMemory_m_avg_stall_time": req_mem_avg_stall,
                "L1_mandatoryQueue_m_stall_time": l1_mand_stall,
                "L1_mandatoryQueue_m_avg_stall_time": l1_mand_avg_stall,
            }

            # include any other parsed keys too
            for k, v in parsed.items():
                if k not in row:
                    row[k] = v

            rows.append(row)

    # finalize df
    df = pd.DataFrame(rows)
    if df.empty:
        print("No stats files found in", results_dir)
        return

    df.sort_values(["workload", "predictor", "ROB_config", "IQ_config"], inplace=True, na_position="last")
    out_csv = os.path.join(out_dir, "detailed_summary_table.csv")
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows, scanned {files_scanned} files)")

    # Write LaTeX table (subset of columns)
    out_tex = os.path.join(out_dir, "detailed_summary_table.tex")
    make_latex_table(df, out_tex)
    print("Wrote LaTeX table:", out_tex)

    # Print a short summary to terminal for quick inspection
    print("\nSample rows (first 8):")
    with pd.option_context('display.max_columns', None):
        print(df.head(8).to_string(index=False))

    print("\nNotes:")
    print("- If avg_recovery_penalty_cycles is -- then no explicit recovery-cycle stat was found.")
    print("- avg_squashed_insts_per_mispred is a useful proxy (squashed instructions per misprediction).")
    print("- If some columns show NaN/--, grep your stats files to find the exact stat name and we can add it to PATTERN_MAP.\n")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=DEFAULT_RESULTS)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    args = p.parse_args()
    main(args.results_dir, args.out_dir)
