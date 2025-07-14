# run_all.py
import os, subprocess, sys
cmds = [
    "python 01_preprocess.py --dataset yoochoose1_64",
    "python 02_global_graph.py --dataset yoochoose1_64",
    "python 04_train_eval.py --dataset yoochoose1_64 --epochs 20"
]
for cmd in cmds:
    print("▶", cmd)
    subprocess.run(cmd.split())