"""Build a CSV describing the SAE features that stand out in the approval task, with Neuronpedia's labels."""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from mortgageapplication.outputs import latest_output, output_path, write_run_info
from mortgageapplication.sae_analysis import ERROR, examples, paired_stats, pivot_by_race

load_dotenv()

DATA_DIR = Path(__file__).parent.parent / "data"
FEATURES_STEM = "approval_sae_features_l20_16k"
PROMPTS_STEM = "approval_sae_prompts_l20_16k"
OUT_STEM = "approval_sae_feature_table_l20_16k"

# Neuronpedia's id for this SAE (model/source); the feature number follows.
NEURONPEDIA_SOURCE = "gemma-2-9b/20-gemmascope-res-16k"
NEURONPEDIA_PAGE = f"https://www.neuronpedia.org/{NEURONPEDIA_SOURCE}/{{feature}}"
NEURONPEDIA_API = f"https://www.neuronpedia.org/api/feature/{NEURONPEDIA_SOURCE}/{{feature}}"
# Raw API responses are kept here so a rerun doesn't fetch again.
CACHE_DIR = DATA_DIR / "neuronpedia_cache"
PAUSE_SECONDS = 0.3

ALL_RACES = ["whi", "bla", "his", "asi", "oth"]
RACES = ["bla", "his", "asi", "oth"]
ROLES = ["name", "city", "last"]
# Features kept per (role, race): the top few by |t| in half A of the sets, on each of two measures.
TOP_PER_GROUP = 15


def role_table(features, role, sets, half_a, half_b):
    """One row per selected feature at this position, with its numbers for every race."""
    act = pivot_by_race(features, role, "act")
    attr = pivot_by_race(features, role, "attr")
    n_sets = len(sets)

    cols = {}
    for race in ALL_RACES:
        cols[f"active_{race}"] = (act[race] > 0).groupby(level="feature").sum() / n_sets
        cols[f"mean_act_{race}"] = act[race].groupby(level="feature").sum() / n_sets
    for race in RACES:
        whole = paired_stats(act, race, sets)
        cols[f"diff_{race}"] = whole["mean"]
        cols[f"t_{race}"] = whole["t"]
        cols[f"t_{race}_A"] = paired_stats(act, race, half_a)["t"]
        cols[f"t_{race}_B"] = paired_stats(act, race, half_b)["t"]
        effect = paired_stats(attr, race, sets)
        cols[f"logit_effect_{race}"] = effect["mean"]
        cols[f"logit_effect_t_{race}"] = effect["t"]
        cols[f"_effect_t_A_{race}"] = paired_stats(attr, race, half_a)["t"]
    table = pd.DataFrame(cols).drop(index=ERROR, errors="ignore")

    # Keep the features that stand out in half A, on activation and on contribution to the decision.
    chosen = {}
    for race in RACES:
        for basis, column in (("activation", f"t_{race}_A"), ("logit_effect", f"_effect_t_A_{race}")):
            for feature in table[column].abs().nlargest(TOP_PER_GROUP).index:
                chosen.setdefault(feature, []).append(f"{basis}:{race}")
    table = table.loc[list(chosen)].drop(columns=[f"_effect_t_A_{r}" for r in RACES])
    table.insert(0, "selected_by", pd.Series({f: "; ".join(v) for f, v in chosen.items()}))

    # The race the feature separates most from white, and whether that holds in both halves of the sets.
    t_whole = table[[f"t_{r}" for r in RACES]].fillna(0)
    best = t_whole.abs().idxmax(axis=1).str.removeprefix("t_")
    direction = [("more" if t_whole.at[f, f"t_{r}"] > 0 else "less") for f, r in best.items()]
    table.insert(1, "strongest", [f"{r} ({d} than white)" for r, d in zip(best, direction)])
    holds = []
    for feature, race in best.items():
        a, b = table.at[feature, f"t_{race}_A"], table.at[feature, f"t_{race}_B"]
        holds.append(bool(a * b > 0 and min(abs(a), abs(b)) >= 3))
    table.insert(2, "replicates", holds)

    table.insert(0, "role", role)
    table["examples"] = [examples(features, role, f) for f in table.index]
    return table.rename_axis("feature").reset_index()


def fetch_feature(feature, session_warnings):
    """Neuronpedia's JSON for one feature, from the cache or the API; None if it can't be fetched."""
    path = CACHE_DIR / f"{feature}.json"
    if path.exists():
        return json.loads(path.read_text())

    headers = {"User-Agent": "feature-table-script"}
    if os.environ.get("NEURONPEDIA_API_KEY"):
        headers["x-api-key"] = os.environ["NEURONPEDIA_API_KEY"]
    request = urllib.request.Request(NEURONPEDIA_API.format(feature=feature), headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            info = json.loads(response.read())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        session_warnings.append(f"feature {feature}: {error}")
        return None

    CACHE_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(info))
    time.sleep(PAUSE_SECONDS)
    return info


def describe(info):
    """The label and a few other fields from a Neuronpedia feature response (missing fields stay blank)."""
    if not info:
        return {}
    explanations = info.get("explanations") or []
    examples_ = info.get("activations") or []
    return {
        "np_label": explanations[0].get("description") if explanations else None,
        "np_top_tokens": " ".join(info.get("pos_str") or [])[:120] or None,
        "np_density": info.get("frac_nonzero"),
        "np_example": "".join(examples_[0].get("tokens") or [])[:120] if examples_ else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-neuronpedia", action="store_true", help="build the table without fetching labels")
    args = parser.parse_args()

    features = pd.read_parquet(latest_output(DATA_DIR, FEATURES_STEM, ".parquet"))
    prompts = pd.read_parquet(latest_output(DATA_DIR, PROMPTS_STEM, ".parquet"))
    features = features.merge(prompts[["prompt_id", "set", "top", "name", "city"]], on="prompt_id")

    sets = sorted(prompts["set"].unique())
    half_a, half_b = sets[0::2], sets[1::2]
    table = pd.concat([role_table(features, role, sets, half_a, half_b) for role in ROLES], ignore_index=True)
    table.insert(2, "url", table["feature"].map(lambda f: NEURONPEDIA_PAGE.format(feature=f)))

    if not args.skip_neuronpedia:
        warnings = []
        info = {f: describe(fetch_feature(f, warnings)) for f in tqdm(table["feature"].unique(), desc="Neuronpedia")}
        labels = pd.DataFrame.from_dict(info, orient="index").rename_axis("feature").reset_index()
        table = table.merge(labels, on="feature", how="left")
        # keep the Neuronpedia columns next to the identifying ones
        front = ["role", "feature", "url", "np_label", "np_top_tokens", "np_density", "np_example"]
        table = table[front + [c for c in table.columns if c not in front]]
        for line in warnings[:5]:
            print(f"warning: {line}")
        if len(warnings) > 5:
            print(f"warning: ... and {len(warnings) - 5} more failed lookups")
        if table["np_label"].isna().all():
            print("warning: no feature has a label; check the field names in the cached responses "
                  f"in {CACHE_DIR}")

    table = table.sort_values(["role", "replicates"], ascending=[True, False]).round(4)
    out_path = output_path(DATA_DIR, OUT_STEM, ".csv")
    table.to_csv(out_path, index=False)
    write_run_info(out_path, script=Path(__file__).name, features=FEATURES_STEM, neuronpedia=NEURONPEDIA_SOURCE,
                   fetched_labels=not args.skip_neuronpedia, top_per_group=TOP_PER_GROUP)
    print(f"{len(table)} rows ({table['feature'].nunique()} distinct features) -> {out_path}")


if __name__ == "__main__":
    main()
