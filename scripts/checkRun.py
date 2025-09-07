#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Paths
GEM5 = os.path.expanduser("~/gem5/build/X86/gem5.opt")
CONFIG = os.path.expanduser("~/Assignment_2/scripts/config.py")

WORK = os.path.expanduser("~/Assignment_2/mibench")
RESULTS = os.path.expanduser("~/Assignment_2/results")
Path(RESULTS).mkdir(parents=True, exist_ok=True)

BPS = ["local", "gshare", "tournament", "bimode"]


workloads = {
    "quicksort": {
        "bin": os.path.join(WORK, "automotive/qsort/qsort_small"),
        "options": os.path.join(WORK, "automotive/qsort/input_small.dat")
    },
    "dijkstra": {
        "bin": os.path.join(WORK, "network/dijkstra/dijkstra_small"),
        "options": os.path.join(WORK, "network/dijkstra/input.dat")
    }
}

for name, wl in workloads.items():
    stats_file = os.path.join(RESULTS, f"stats_{name}.txt")
    bina = wl["bin"]
    opt = wl["options"]

    for bp in BPS:
        stats_file = os.path.join(RESULTS, f"stats_{name}_{bp}.txt")

        print(f"\n=== Running {name} with BP={bp} ===")
        print(f"[INFO] Stats -> {stats_file}")

        cmd = [
            GEM5,
            f"--stats-file={stats_file}",
            CONFIG,
            "--cmd", bina,
            "--options", opt,
            "--bp", bp
        ]

        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)


    # print(f"\n=== Running {name} ===")
    # print(f"[INFO] Stats -> {stats_file}")
    # cmd = [
    #     GEM5,
    #     f"--stats-file={stats_file}",
    #     CONFIG,
    #     "--cmd", bina,            # <-- binary
    #     "--options", opt,         # <-- input file as string
    #     "--bp", BP          
    # ]
    # print("Running:", " ".join(cmd))

    # subprocess.run(cmd, check=True)

print("\n[INFO] All runs complete. Stats files saved in", RESULTS)
