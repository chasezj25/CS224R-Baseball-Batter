"""
filter_data.py

Reads the swing metadata CSV and removes left-handed hitters, writing the
remaining rows to eligible_swings.csv for use in downstream preprocessing.
"""

import pandas as pd

# Only right-handed swings are supported in this pipeline
FILTER_COLUMN = "hitter_side"
FILTER_VALUE = "L"


def main():
    df = pd.read_csv("../data/data/metadata.csv")
    df = df[df[FILTER_COLUMN] != FILTER_VALUE]
    df.to_csv("../eligible_swings.csv", index=False)
    print(f"Kept {len(df)} right-handed swings.")


if __name__ == "__main__":
    main()
