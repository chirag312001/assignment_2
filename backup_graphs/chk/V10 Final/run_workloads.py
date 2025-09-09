#!/usr/bin/env python3
import os, subprocess, csv, time

# Paths (adjust if needed)
GEM5 = os.path.expanduser("~/gem5/build/X86/gem5.opt")
SE = os.path.expanduser("~/gem5/configs/deprecated/example/se.py")
WORK = os.path.expanduser("~/Assignment_2")
RESULTS = os.path.join(WORK, "results", "python_runs")

# Workloads (update paths if you built quicksort elsewhere)
workloads = {
    "quicksort": {
        "bin": os.path.join(WORK, "mibench/automotive/qsort/qsort_small"),
        "options": os.path.join(WORK, "mibench/automotive/qsort/input_small.dat")
    },
    "dijkstra": {
        "bin": os.path.join(WORK, "mibench/network/dijkstra/dijkstra_small"),
        "options": os.path.join(WORK, "mibench/network/dijkstra/input.dat")
    }
}

predictors = ["BiModeBP", "GshareBP", "LocalBP", "TournamentBP"]
MAXINSTS = 20000000

os.makedirs(RESULTS, exist_ok=True)
summary_file = os.path.join(RESULTS, "summary.csv")

with open(summary_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["workload", "predictor", "simInsts", "numCycles", "IPC",
                     "condPredicted", "condIncorrect", "mispredictPercent", "status"])

    for wl_name, wl_info in workloads.items():
        binpath = wl_info["bin"]
        if not os.path.isfile(binpath) or not os.access(binpath, os.X_OK):
            print(f"[WARN] Skipping {wl_name}: binary not found or not executable at {binpath}")
            continue

        for pred in predictors:
            outdir = os.path.join(RESULTS, f"{wl_name}_{pred.lower()}")
            os.makedirs(outdir, exist_ok=True)

            cmd = [GEM5, "-d", outdir, SE,
                   "--cpu-type=DerivO3CPU", "--caches", "--l2cache",
                   f"--bp-type={pred}", "--maxinsts", str(MAXINSTS),
                   "-c", binpath]
            if wl_info["options"]:
                cmd += ["--options", wl_info["options"]]

            print(f"\n[{time.strftime('%H:%M:%S')}] Running {wl_name} with {pred} ...")
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                status = "OK"
            except subprocess.CalledProcessError as e:
                status = f"FAILED({e.returncode})"
                # print a few lines of messages for debugging
                msgfile = os.path.join(outdir, "messages")
                if os.path.isfile(msgfile):
                    print("=== tail messages ===")
                    try:
                        print("\n".join(open(msgfile).read().splitlines()[-40:]))
                    except Exception:
                        pass
                else:
                    print("[INFO] no messages file found (gem5 may have failed before creating it).")

            # parse stats if present
            stats_file = os.path.join(outdir, "stats.txt")
            simInsts = numCycles = condPred = condIncorrect = 0
            if os.path.isfile(stats_file):
                with open(stats_file) as f:
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            key = parts[0]
                            val = parts[1]
                            if key == "simInsts":
                                simInsts = int(val)
                            elif key == "numCycles":
                                numCycles = int(val)
                            elif "bpred.condPredicted" in key or "condPredicted" in key:
                                condPred = int(val)
                            elif "bpred.condIncorrect" in key or "condIncorrect" in key:
                                condIncorrect = int(val)
            IPC = (simInsts / numCycles) if numCycles > 0 else 0.0
            misp = (condIncorrect / condPred * 100.0) if condPred > 0 else 0.0

            writer.writerow([wl_name, pred, simInsts, numCycles, f"{IPC:.6f}",
                             condPred, condIncorrect, f"{misp:.6f}", status])

print("All done. Summary at:", summary_file)
