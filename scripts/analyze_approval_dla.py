from pathlib import Path
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATA_DIR = Path(__file__).parent.parent / "data"
DLA_PATH = DATA_DIR / "approval_dla_gemma-2-9b.parquet"

RACES = ["whi", "bla", "his", "asi", "oth"]


def layer_table(rows):
    """One row per layer, columns attn and mlp: how much each pushes approve over review."""
    parts = rows["component"].str.extract(r"^(?P<layer>\d+)_(?P<kind>attn|mlp)_out$")
    kept = rows.assign(layer=parts["layer"], kind=parts["kind"]).dropna(subset=["layer"])
    kept = kept.assign(layer=kept["layer"].astype(int))
    return kept.pivot(index="layer", columns="kind", values="dla").sort_index()


def plot_layers(tables, title):
    """Heatmaps of per-layer attention and MLP contributions, one panel per race, on a shared scale."""
    races = [r for r in RACES if r in tables]
    scale = max(np.abs(t.to_numpy()).max() for t in tables.values())

    fig = make_subplots(rows=1, cols=len(races), shared_yaxes=True,
                        horizontal_spacing=0.02, subplot_titles=races)
    for col, race in enumerate(races, start=1):
        table = tables[race]
        fig.add_trace(go.Heatmap(
            z=table.to_numpy(), x=["attn", "mlp"], y=table.index,
            colorscale="RdBu", zmid=0, zmin=-scale, zmax=scale,
            showscale=col == len(races), colorbar={"title": "Δ logit"},
            hovertemplate="layer %{y} %{x}: %{z:.3f}<extra></extra>"), 1, col)

    fig.update_yaxes(autorange="reversed", dtick=2, title_text="Layer", col=1)
    fig.update_layout(title=title, width=1000, height=800)
    fig.show()


def main():
    frame = pd.read_parquet(DLA_PATH)

    print("logit difference (approve - review) by race:")
    print(frame.groupby("top")["logit_diff"].first().reindex(RACES).round(3))

    tables = {race: layer_table(rows) for race, rows in frame.groupby("top")}

    # Where the races differ, relative to the white version of the same listing.
    diffs = {race: tables[race] - tables["whi"] for race in tables if race != "whi"}
    print("\nlargest layer contributions to the gap vs white:")
    for race, diff in diffs.items():
        top = diff.stack().sort_values(key=np.abs, ascending=False).head(3)
        print(f"  {race}: " + ", ".join(f"L{layer} {kind} {value:+.2f}" for (layer, kind), value in top.items()))

    plot_layers(tables, "Direct logit attribution: approve − review")
    plot_layers(diffs, "Difference from the white version")


if __name__ == "__main__":
    main()
