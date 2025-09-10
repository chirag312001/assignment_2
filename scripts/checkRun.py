#!/usr/bin/env python3
import os

import subprocess
from pathlib import Path

# Paths
GEM5 = os.path.expanduser("~/gem5/build/X86/gem5.opt")
CONFIG = os.path.expanduser("~/Assignment_2/scripts/config.py")
WORK = os.path.expanduser("~/Assignment_2/mibench")
RESULTS = os.path.expanduser("~/Assignment_2/create")
Path(RESULTS).mkdir(parents=True, exist_ok=True)


# defining variables
BPS = ["local", "gshare", "tournament", "bimode"]

rob_vals = [128, 256]
num_IQentres = [64,128]

# defining workload
workloads = {
    "quicksort_small": {
        "bin": os.path.join(WORK, "automotive/qsort/qsort_small"),
        "options": os.path.join(WORK, "automotive/qsort/input_small.dat")
    },
    "quicksort_large": {
        "bin": os.path.join(WORK, "automotive/qsort/qsort_large"),
        "options": os.path.join(WORK, "automotive/qsort/input_large.dat")
    },
    "dijkstra_small": {
        "bin": os.path.join(WORK, "network/dijkstra/dijkstra_small"),
        "options": os.path.join(WORK, "network/dijkstra/input.dat")
    },
    # "dijkstra_large": {
    #     "bin": os.path.join(WORK, "network/dijkstra/dijkstra_large"),
    #     "options": os.path.join(WORK, "network/dijkstra/input.dat")
    # },
    "basicmath":{
        "bin":os.path.join(WORK, "automotive/basicmath/basicmath_small"),
        "options":"s"
    },
    # "basicmath":{
    #     "bin":os.path.join(WORK, "automotive/basicmath/basicmath_large"),
    #     "options":"s"
    # }
    "fft":{
        "bin":os.path.join(WORK, "telecomm/FFT/fft"),
        "options": "s"
    }

}


# running for each variable and workload

for name, wl in workloads.items():

    stats_file = os.path.join(RESULTS, f"stats_{name}.txt")
    bina = wl["bin"]
    opt = wl["options"]

    for bp in BPS:
        for rob in rob_vals:
            for iq in num_IQentres:
                stats_file = os.path.join(RESULTS, f"stats_{name}_{bp}_ROB{rob}_numIQEntries{iq}.txt")

                print(f"\n=== Running {name} with BP={bp} , ROB={rob} , IQ={iq}===")
                print(f"[INFO] Stats -> {stats_file}")

                cmd = [
                    GEM5,
                    f"--stats-file={stats_file}",
                    CONFIG,
                    "--cmd", bina,
                    "--options", opt,
                    "--bp", bp,
                    "--rob",str(rob),
                    "--iq",str(iq)
                ]
                #calling 
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
