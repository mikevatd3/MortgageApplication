from pathlib import Path
import pandas as pd

from mortgageapplication.outputs import latest_output


CHALLENGE = "challenge_tables_2026-10-23_full"
RESULT_STEM = "rankings__completion"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE


def main():
    frame = pd.read_csv(latest_output(DATA_DIR, RESULT_STEM, ".csv"))
    frame = frame[frame["valid"]].copy()

    # ranked_rows lists original row positions, best first; the target's rank is where its row appears.
    frame["rank"] = [
        r.split(",").index(str(p))
        for p, r in zip(frame["position"], frame["ranked_rows"])
    ]
    # Negative change = moved toward the front (more favorable).
    frame["change"] = frame["rank"] - frame["position"]
    frame["relative_change"] = frame["change"] - frame["change"].mean()

    print(f"{len(frame)} valid rankings")
    print(frame.groupby("race")[["rank", "change", "relative_change"]].mean().round(3))


if __name__ == "__main__":
    main()
