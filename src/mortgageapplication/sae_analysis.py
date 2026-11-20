"""Shared pieces of the SAE feature analyses: paired differences against the white version, and examples."""
import numpy as np
import pandas as pd

# The SAE's reconstruction error is stored as a pseudo-feature with this id.
ERROR = -1


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


def examples(features, role, feature, n=3):
    """The inputs where a feature fires hardest, as "text [race]"."""
    rows = features[(features["role"] == role) & (features["feature"] == feature)]
    text = "city" if role == "city" else "name"
    return ", ".join(f"{r[text]} [{r['top']}]" for _, r in rows.nlargest(n, "act").iterrows())
