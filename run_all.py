#!/usr/bin/env python3
"""Run every step in order and keep the complete terminal output in out/train_log.txt.

    python run_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = (['fit_classical.py'], ['train_residual.py'], ['train_residual.py', '--hidden', '64', '64'],
         ['evaluate.py'], ['export_model.py'], ['verify.py'])

(HERE / 'out').mkdir(exist_ok=True)
with open(HERE / 'out' / 'train_log.txt', 'w') as log:
    for step in STEPS:
        banner = f'\n===== python {" ".join(step)} =====\n'
        print(banner, end='', flush=True); log.write(banner)
        process = subprocess.Popen([sys.executable, '-u', *step], cwd=HERE, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True); log.write(line); log.flush()
        if process.wait() != 0:
            sys.exit(f'step failed: {" ".join(step)} (see out/train_log.txt)')
