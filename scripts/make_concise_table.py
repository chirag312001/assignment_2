#!/usr/bin/env python3
# scripts/make_concise_table.py
import pandas as pd, numpy as np, os
infile = "analysis/detailed_summary_table.csv"
outcsv = "analysis/concise_report_table.csv"
outtex = "analysis/concise_report_table.tex"
df = pd.read_csv(infile)
# keep key columns and format numbers
keep = ["workload","predictor","ROB_config","IQ_config","ipc","branch_accuracy","mispred_rate","avg_recovery_penalty_cycles","avg_squashed_insts_per_mispred","L1D_accesses","L1D_misses"]
exist = [c for c in keep if c in df.columns]
short = df[exist].copy()
short["branch_accuracy_pct"] = (short["branch_accuracy"]*100).round(3)
short["mispred_rate_pct"] = (short["mispred_rate"]*100).round(3)
short["ipc"] = short["ipc"].round(4)
cols_out = ["workload","predictor","ROB_config","IQ_config","ipc","branch_accuracy_pct","mispred_rate_pct","avg_recovery_penalty_cycles","avg_squashed_insts_per_mispred","L1D_accesses","L1D_misses"]
short.to_csv(outcsv, index=False, columns=cols_out)
# LaTeX
with open(outtex,"w") as f:
    f.write(short.to_latex(index=False,columns=cols_out,float_format="%.4g"))
print("Wrote", outcsv, outtex)
