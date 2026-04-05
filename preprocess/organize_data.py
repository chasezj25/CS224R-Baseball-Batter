"""
organize_data.py

Runs the full preprocessing pipeline in order:
  1. filter_data   – filters to right-handed swings → eligible_swings.csv
  2. gen_bat_data  – extracts bat trajectories → bat_data.pkl / bat_data_100hz.pkl
  3. sort_data     – merges joint angles + velocities → sorted_data.pkl
"""

import runpy
import os

SCRIPTS = [
    "filter_data.py",
    "gen_bat_data.py",
    "sort_data.py",
]


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for script in SCRIPTS:
        path = os.path.join(script_dir, script)
        print(f"Running {script}...")
        runpy.run_path(path, run_name="__main__")
        print(f"Finished {script}.\n")


if __name__ == "__main__":
    main()
