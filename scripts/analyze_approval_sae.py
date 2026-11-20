from pathlib import Path
import numpy as np
import pandas as pd

from mortgageapplication.outputs import latest_output


DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_STEM = "approval_sae_features_l20_16k"
PROMPTS_STEM = "approval_sae_prompts_l20_16k"
NEURONPEDIA = "https://www.neuronpedia.org/gemma-2-9b/20-gemmascope-res-16k/{feature}"

RACES = ["bla", "his", "asi", "oth"]
ROLES = ["name", "city", "last"]
ERROR = -1
TOP = 8


def pivot_by_race(features, role, value):
    """One row per (set, feature), one column per race, 0 where the feature wasn't active."""
    rows = features[features["role"] == role]
    return rows.pivot_table(index=["set", "feature"], columns="top", values=value, aggfunc="sum", fill_value=0)


def paired_stats(wide, race, sets):
    """Per feature: the mean over `sets` of (race - white), its standard error, and t.

    A feature missing from a set was inactive in all five versions, so it counts as a difference of 0.
    """
    wide = wide[wide.index.get_level_values("set").isin(sets)]
    diff = wide[race] - wide["whi"]
    n = len(sets)
    mean = diff.groupby(level="feature").sum() / n
    var = ((diff ** 2).groupby(level="feature").sum() - n * mean ** 2) / (n - 1)
    se = np.sqrt(var.clip(lower=0) / n)
    return pd.DataFrame({"mean": mean, "se": se, "t": mean / se.replace(0, np.nan)})


def ranked(wide, act, race, half_a, half_b, n_sets):
    """The features with the largest paired difference in half A of the sets, with how they do in half B."""
    a = paired_stats(wide, race, half_a)
    b = paired_stats(wide, race, half_b)
    table = pd.DataFrame({"mean_A": a["mean"], "t_A": a["t"], "mean_B": b["mean"], "t_B": b["t"]})
    table = table.drop(index=ERROR, errors="ignore")
    table = table.assign(abs_t=table["t_A"].abs()).sort_values("abs_t", ascending=False).drop(columns="abs_t").head(TOP)

    active_race = (act[race] > 0).groupby(level="feature").sum() / n_sets
    active_white = (act["whi"] > 0).groupby(level="feature").sum() / n_sets
    table["active_race"] = active_race.reindex(table.index)
    table["active_white"] = active_white.reindex(table.index)
    return table


def examples(features, role, feature, n=3):
    """The inputs where a feature fires hardest, as "text [race]"."""
    rows = features[(features["role"] == role) & (features["feature"] == feature)]
    text = "city" if role == "city" else "name"
    return ", ".join(f"{r[text]} [{r['top']}]" for _, r in rows.nlargest(n, "act").iterrows())


def show_ranked(title, features, value, act_by_role):
    half_a = sorted(features["set"].unique())[0::2]
    half_b = sorted(features["set"].unique())[1::2]
    n_sets = features["set"].nunique()

    print(f"\n{title}")
    print("features ranked by |t| in half A of the sets (even-numbered), checked in half B (odd-numbered)")
    for role in ROLES:
        wide = pivot_by_race(features, role, value)
        for race in RACES:
            table = ranked(wide, act_by_role[role], race, half_a, half_b, n_sets)
            print(f"\n[{role} tokens, {race} vs whi]")
            print(table.round(3).to_string())
            for feature in table.index[:3]:
                print(f"  {feature}: {examples(features, role, feature)}  {NEURONPEDIA.format(feature=feature)}")


def main():
    features = pd.read_parquet(latest_output(DATA_DIR, FEATURES_STEM, ".parquet"))
    prompts = pd.read_parquet(latest_output(DATA_DIR, PROMPTS_STEM, ".parquet"))
    features = features.merge(prompts[["prompt_id", "set", "top", "name", "city"]], on="prompt_id")

    n_sets = prompts["set"].nunique()
    print(f"{len(prompts)} prompts, {n_sets} sets")

    print("\nhow well the SAE reconstructs the residual (error is the share of its squared norm missed):")
    fidelity = pd.DataFrame({
        role: {"mean_error": prompts[f"{role}_recon_error"].mean(), "mean_active_features": prompts[f"{role}_l0"].mean()}
        for role in ROLES
    })
    print(fidelity.round(3).to_string())

    act_by_role = {role: pivot_by_race(features, role, "act") for role in ROLES}

    # Step 1: which features differ between a race's version and the white version.
    show_ranked("STEP 1: difference in feature activation (summed over the span's tokens)",
                features, "act", act_by_role)

    # Step 2: which features account for the difference in the model's decision.
    show_ranked("STEP 2: difference in each feature's contribution to the logit difference (activation × gradient)",
                features, "attr", act_by_role)

    # How much of the gap in the decision each position's features and error term carry (first order).
    logit_diff = prompts.pivot(index="set", columns="top", values="logit_diff")
    rows = []
    for race in RACES:
        row = {"race": race, "logit_gap": (logit_diff[race] - logit_diff["whi"]).mean()}
        for role in ROLES:
            wide = pivot_by_race(features, role, "attr")
            diff = wide[race] - wide["whi"]
            is_error = diff.index.get_level_values("feature") == ERROR
            row[f"{role}_features"] = diff[~is_error].groupby(level="set").sum().reindex(logit_diff.index, fill_value=0).mean()
            row[f"{role}_error"] = diff[is_error].groupby(level="set").sum().reindex(logit_diff.index, fill_value=0).mean()
        rows.append(row)
    print("\nmean gap in the logit difference vs white, and the first-order contribution of each position's")
    print("features and of the SAE's error term (these needn't add up to the gap):")
    print(pd.DataFrame(rows).set_index("race").round(3).to_string())


if __name__ == "__main__":
    main()
