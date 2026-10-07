"""
run_all.py
==========
Master script to run the complete Phase 1 pipeline:

Objective 1: Dataset Integration and Cleaning
  - src/integrate_data.py
  - src/clean_data.py

Objective 2: AI/ML Model Training and Evaluation
  - src/train_model.py
  - src/evaluate_model.py
"""

import subprocess
import sys

scripts = [
    "src/integrate_data.py",
    "src/clean_data.py",
    "src/train_model.py",
    "src/evaluate_model.py",
]

if __name__ == "__main__":
    for script in scripts:
        print(f"\n>>> Running {script}...")
        result = subprocess.run([sys.executable, script])
        if result.returncode != 0:
            print(f"ERROR: {script} failed with exit code {result.returncode}")
            sys.exit(result.returncode)
    print("\n>>> All pipeline steps completed successfully.")
