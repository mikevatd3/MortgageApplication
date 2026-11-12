from pathlib import Path
import numpy as np
import pandas as pd


CHALLENGE = "challenge_tables_2026-10-23_full"
RESULT_NAME = "approvals__completion.csv"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE

RACES = ["whi", "bla", "his", "asi", "oth"]


def paired_vs_white(frame, value):
    """Per-race mean of (value - the same batch's white value), with its standard error."""
    wide = frame.pivot(index="batch", columns="race", values=value)
    diff = wide.sub(wide["whi"], axis=0).dropna(subset=["whi"])
    n = diff.count()
    return pd.DataFrame({
        "mean": wide.mean(),
        "diff_vs_white": diff.mean(),
        "se": diff.std() / np.sqrt(n),
        "share_differing": (diff.abs() > 1e-9).sum() / n,
        "batches": n,
    }).reindex(RACES)


def main():
    frame = pd.read_csv(DATA_DIR / RESULT_NAME)

    # Positive = leans toward "approve" over "review".
    frame["log_odds"] = frame["logprob_approve"] - frame["logprob_review"]
    # How much of the model's next-token probability lands on the two decision words at all.
    frame["pair_mass"] = np.exp(frame["logprob_approve"]) + np.exp(frame["logprob_review"])

    print(f"{len(frame)} rows")

    for variant, rows in frame.groupby("variant", sort=False):
        print(f"\n== {variant} ==")
        print(f"mean probability on approve/review: {rows['pair_mass'].mean():.3f}")
        print(f"most common top tokens: {rows['top_token'].value_counts().head(3).to_dict()}")
        print("log-odds of approve vs review; diff_vs_white > 0 means more approving than the white version")
        print(paired_vs_white(rows, "log_odds").round(3))


if __name__ == "__main__":
    main()
