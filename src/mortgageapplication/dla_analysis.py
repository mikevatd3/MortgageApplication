"""Shared pieces of the direct-logit-attribution analyses: layer tables, paired-vs-white, and heatmaps."""
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

RACES = ["whi", "bla", "his", "asi", "oth"]


def layer_rows(frame):
    """The per-layer attention and MLP rows, with `layer` and `kind` columns (embed is dropped)."""
    parts = frame["component"].str.extract(r"^(?P<layer>\d+)_(?P<kind>attn|mlp)_out$")
    kept = frame.assign(layer=parts["layer"], kind=parts["kind"]).dropna(subset=["layer"])
    return kept.assign(layer=kept["layer"].astype(int))


def paired_vs_white(wide):
    """Per-race mean of (value - the same unit's white value), with its standard error across units.

    `wide` has one row per unit (a set or a batch) and one column per race.
    """
    diff = wide.sub(wide["whi"], axis=0).dropna(subset=["whi"])
    n = diff.count()
    return pd.DataFrame({
        "mean": wide.mean(),
        "diff_vs_white": diff.mean(),
        "se": diff.std() / np.sqrt(n),
        "n": n,
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
