from pathlib import Path
import numpy as np
import pandas as pd

from mortgageapplication.dla_analysis import layer_rows, paired_vs_white, plot_layers
from mortgageapplication.outputs import latest_output


DATA_DIR = Path(__file__).parent.parent / "data"
DLA_STEM = "approval_dla_gemma-2-9b"


def leaning(logit_diff):
    """The decision the model leans toward, and how much of the approve/review pair's probability it gets."""
    p_approve = 1 / (1 + np.exp(-logit_diff))
    word, share = ("approve", p_approve) if logit_diff >= 0 else ("review", 1 - p_approve)
    return f"{word} {share:.1%}"


def main():
    frame = pd.read_parquet(latest_output(DATA_DIR, DLA_STEM, ".parquet"))

    # One logit difference per (set, race): positive = leans toward "approve".
    decisions = frame.groupby(["set", "top"])["logit_diff"].first().unstack("top")
    print(f"{len(decisions)} sets")

    summary = paired_vs_white(decisions)
    summary["leans"] = summary["mean"].map(leaning)
    summary["share_approve"] = (decisions > 0).mean().reindex(summary.index)
    print("\nlogit difference (approve - review) by race, averaged over sets:")
    print(summary.round(3))

    # Per-layer contributions: each race against white within the same set, then averaged over sets.
    layers = layer_rows(frame)
    wide = layers.pivot_table(index=["set", "layer", "kind"], columns="top", values="dla")
    diff = wide.sub(wide["whi"], axis=0)
    cells = ["layer", "kind"]

    mean_dla = wide.groupby(level=cells).mean()
    mean_diff = diff.groupby(level=cells).mean()
    se_diff = diff.groupby(level=cells).sem()

    print("\nlargest layer contributions to the gap vs white (mean ± se over sets):")
    for race in [r for r in summary.index if r != "whi"]:
        strongest = (mean_diff[race] / se_diff[race]).abs().sort_values(ascending=False).head(3).index
        print(f"  {race}: " + ", ".join(
            f"L{layer} {kind} {mean_diff.loc[(layer, kind), race]:+.2f} ± {se_diff.loc[(layer, kind), race]:.2f}"
            for layer, kind in strongest))

    own_titles = {r: f"{r}: {leaning(summary.loc[r, 'mean'])}" for r in summary.index}
    diff_titles = {r: f"{r} vs whi: {summary.loc[r, 'diff_vs_white']:+.2f} ± {summary.loc[r, 'se']:.2f}"
                   for r in summary.index if r != "whi"}

    plot_layers({r: mean_dla[r].unstack("kind") for r in summary.index},
                "Direct logit attribution: approve − review (mean over sets)", own_titles)
    plot_layers({r: mean_diff[r].unstack("kind") for r in diff_titles},
                "Difference from the white version (mean over sets)", diff_titles)


if __name__ == "__main__":
    main()
