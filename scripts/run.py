#!/usr/bin/env python3
import os
from config import run_workload

WORK = os.path.expanduser("~/Assignment_2/mibench")
RESULTS = os.path.expanduser("~/Assignment_2/results")

workloads = {
    "quicksort": {
        "bin": os.path.join(WORK, "automotive/qsort/qsort_small"),
        "options": [os.path.join(WORK, "automotive/qsort/input_small.dat")]
    },
    "dijkstra": {
        "bin": os.path.join(WORK, "network/dijkstra/dijkstra_small"),
        "options": [os.path.join(WORK, "network/dijkstra/input.dat")]
    }
}

for name, wl in workloads.items():
    
    print(f"\n=== Running {name} ===")
    run_workload(wl["bin"], wl["options"])
