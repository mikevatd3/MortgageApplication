from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATA_DIR = Path(__file__).parent.parent / "data"
DLA_PATH = DATA_DIR / "approval_dla_gemma-2-9b.parquet"

RACES = ["whi", "bla", "his", "asi", "oth"]


def layer_rows(frame):
    """The per-layer attention and MLP rows, with `layer` and `kind` columns (embed is dropped)."""
    parts = frame["component"].str.extract(r"^(?P<layer>\d+)_(?P<kind>attn|mlp)_out$")
    kept = frame.assign(layer=parts["layer"], kind=parts["kind"]).dropna(subset=["layer"])
    return kept.assign(layer=kept["layer"].astype(int))


def leaning(logit_diff):
    """The decision the model leans toward, and how much of the approve/review pair's probability it gets."""
    p_approve = 1 / (1 + np.exp(-logit_diff))
    word, share = ("approve", p_approve) if logit_diff >= 0 else ("review", 1 - p_approve)
    return f"{word} {share:.1%}"


def paired_vs_white(wide):
    """Per-race mean of (value - the same set's white value), with its standard error across sets."""
    diff = wide.sub(wide["whi"], axis=0).dropna(subset=["whi"])
    n = diff.count()
    return pd.DataFrame({
        "mean": wide.mean(),
        "diff_vs_white": diff.mean(),
        "se": diff.std() / np.sqrt(n),
        "sets": n,
    }).reindex([r for r in RACES if r in wide.columns])


def plot_layers(tables, title, panel_titles):
    """Heatmaps of per-layer attention and MLP contributions, one panel per race, on a shared scale."""
    races = [r for r in RACES if r in tables]
    scale = max(np.abs(t.to_numpy()).max() for t in tables.values())

    fig = make_subplots(rows=1, cols=len(races), shared_yaxes=True,
                        horizontal_spacing=0.02, subplot_titles=[panel_titles[r] for r in races])
    for col, race in enumerate(races, start=1):
        table = tables[race]
        fig.add_trace(go.Heatmap(
            z=table.to_numpy(), x=["attn", "mlp"], y=table.index,
            colorscale="RdBu", zmid=0, zmin=-scale, zmax=scale,
            showscale=col == len(races), colorbar={"title": "Δ logit"},
            hovertemplate="layer %{y} %{x}: %{z:.3f}<extra></extra>"), 1, col)

    fig.update_yaxes(autorange="reversed", dtick=2, title_text="Layer", col=1)
    fig.update_annotations(font_size=11)   # the panel titles
    fig.update_layout(title=title, width=1000, height=800)
    fig.show()


def main():
    frame = pd.read_parquet(DLA_PATH)

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
