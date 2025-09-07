import subprocess
from pathlib import Path

gem5_bin = "build/X86/gem5.opt"
config_script = "hello-world.py"
csv_file = Path("results.csv")

# Write CSV header once
with open(csv_file, "w") as f:
    f.write("ticks,time_ms,freq_MHz,ram,cpu_type\n")

cpu_list = ["ATOMIC", "TIMING", "MINOR", "O3"]
freq_list = range(600, 3400, 200)
mem_list = ["DDR3_1600", "DDR4_2400"]

for cpu in cpu_list:
    for freq in freq_list:
        for mem in mem_list:
            print(f"[RUN] {cpu} {freq}MHz {mem}")
            subprocess.run(
                [gem5_bin, config_script, cpu, str(freq), mem, str(csv_file)],
                check=True
            )
