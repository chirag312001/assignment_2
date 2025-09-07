#!/bin/bash
set -euo pipefail

# Paths
GEM5="$HOME/gem5/build/X86/gem5.opt"
CONFIG="$HOME/Assignment_2/scripts/config.py"
WORK="$HOME/Assignment_2/mibench"
RESULTS="$HOME/Assignment_2/results"

mkdir -p "$RESULTS"

# Workloads
# You can add more here if needed
declare -A bins
declare -A opts

bins[dijkstra]="$WORK/network/dijkstra/dijkstra_small"
opts[dijkstra]="$WORK/network/dijkstra/input.dat"

# Run workloads
for name in "${!bins[@]}"; do
    bina="${bins[$name]}"
    opt="${opts[$name]}"
    stats_file="$RESULTS/stats_${name}.txt"
    logfile="$RESULTS/gem5_${name}.log"

    echo ""
    echo "=== Running $name ==="
    echo "[INFO] Stats -> $stats_file"
    echo "Running: $GEM5 --stats-file=$stats_file $CONFIG $bina $opt"

    "$GEM5" --stats-file="$stats_file" "$CONFIG" "$bina" "$opt" \
        >"$logfile" 2>&1

    echo "[OK] Finished $name (log -> $logfile)"
done
