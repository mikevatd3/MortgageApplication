from pathlib import Path
import pandas as pd

from mortgageapplication.dla_analysis import layer_rows, paired_vs_white, plot_layers
from mortgageapplication.outputs import latest_output


CHALLENGE = "challenge_tables_2026-11-20"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
DLA_STEM = "selection_dla_gemma-2-9b"


def contrast(frame, value):
    """Per prompt: the test applicant's value minus the mean of the four rivals', for the tokens' `value` column."""
    people = frame[frame["role"] != "white_baseline"]
    side = people.pivot_table(index=["batch", "race", "fileid"], columns="role", values=value, aggfunc="mean")
    return (side["test"] - side["rival"]).rename("score")


def main():
    frame = pd.read_parquet(latest_output(DATA_DIR, DLA_STEM, ".parquet"))

    prompts = frame.drop_duplicates("fileid")
    print(f"{len(prompts)} prompts, {prompts['collision'].sum()} with two applicants sharing a first token")
    print(f"most common top tokens: {prompts['top_token'].value_counts().head(3).to_dict()}")

    # Score = the test applicant's logit minus the rivals' mean logit; positive = favored over the others.
    # One value per (batch, race), compared with the white version of the same batch.
    people = frame[frame["role"] != "white_baseline"].drop_duplicates(subset=["fileid", "row"])
    score = contrast(people, "logit").reset_index()
    wide_score = score.pivot(index="batch", columns="race", values="score")

    # Is the test applicant the one the model would pick (highest first-token logit of the five)?
    picked = people.loc[people.groupby("fileid")["logit"].idxmax()].set_index("fileid")
    top_choice = (picked["role"] == "test").groupby(picked["race"]).mean()

    summary = paired_vs_white(wide_score)
    summary["top_choice"] = top_choice.reindex(summary.index)
    summary["candidate_mass"] = prompts.groupby("race")["candidate_mass"].mean().reindex(summary.index)
    print("\ntest applicant's logit minus the rivals' mean, by race (chance of being the top choice is 0.2):")
    print(summary.round(3))

    # Per-layer contributions to that score, each race against white within the same batch.
    layers = layer_rows(frame[frame["role"] != "white_baseline"])
    per_component = layers.pivot_table(index=["batch", "race", "fileid", "layer", "kind"],
                                       columns="role", values="dla", aggfunc="mean")
    per_component = (per_component["test"] - per_component["rival"]).rename("score").reset_index()
    wide = per_component.pivot_table(index=["batch", "layer", "kind"], columns="race", values="score")
    diff = wide.sub(wide["whi"], axis=0)
    cells = ["layer", "kind"]

    mean_dla = wide.groupby(level=cells).mean()
    mean_diff = diff.groupby(level=cells).mean()
    se_diff = diff.groupby(level=cells).sem()

    print("\nlargest layer contributions to the gap vs white (mean ± se over batches):")
    for race in [r for r in summary.index if r != "whi"]:
        strongest = (mean_diff[race] / se_diff[race]).abs().sort_values(ascending=False).head(3).index
        print(f"  {race}: " + ", ".join(
            f"L{layer} {kind} {mean_diff.loc[(layer, kind), race]:+.2f} ± {se_diff.loc[(layer, kind), race]:.2f}"
            for layer, kind in strongest))

    own_titles = {r: f"{r}: top choice {summary.loc[r, 'top_choice']:.0%}" for r in summary.index}
    diff_titles = {r: f"{r} vs whi: {summary.loc[r, 'diff_vs_white']:+.2f} ± {summary.loc[r, 'se']:.2f}"
                   for r in summary.index if r != "whi"}

    plot_layers({r: mean_dla[r].unstack("kind") for r in summary.index},
                "Direct logit attribution: test applicant − rivals (mean over batches)", own_titles)
    plot_layers({r: mean_diff[r].unstack("kind") for r in diff_titles},
                "Difference from the white version (mean over batches)", diff_titles)


if __name__ == "__main__":
    main()
