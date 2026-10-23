from pathlib import Path
import pandas as pd


CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE


def find_index(frame):
    frame = frame.copy()

    frame["split_rankings"] = frame["ranking"].str.split(",")

    frame = frame[~frame["split_rankings"].isna()]

    return pd.Series([
        r.index(str(v)) 
        for v, r in zip(frame["position"], frame["split_rankings"])
    ])
    


def main():
    frame = pd.read_csv(DATA_DIR / "rankings__city_only.csv")

    frame["outcome"] = find_index(frame)

    print(frame.groupby("race").outcome.mean())
    

if __name__ == "__main__":
    main()
