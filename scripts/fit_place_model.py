from pathlib import Path
import numpy as np
import pandas as pd

from mortgageapplication.outputs import latest_output, output_path, write_run_info

DATA_DIR = Path(__file__).parent.parent / "data"
ACTS_STEM = "place_activations_gemma-2-9b"
OUT_STEM = "place_model_predictions"

FOLDS = 5
SEED = 0


def cross_val_ols(X, y, folds=FOLDS, seed=SEED):
    """Out-of-fold OLS predictions and R² (each row predicted by a model that never saw it)."""
    idx = np.random.default_rng(seed).permutation(len(y))
    pred = np.zeros(len(y))
    for held_out in np.array_split(idx, folds):
        train = np.setdiff1d(idx, held_out)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
        Xs = (X - mu) / sd
        w = np.linalg.lstsq(Xs[train], y[train] - y[train].mean(), rcond=None)[0]
        pred[held_out] = Xs[held_out] @ w + y[train].mean()
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return pred, r2


def main():
    acts_path = latest_output(DATA_DIR, ACTS_STEM, ".npz")
    data = np.load(acts_path)
    X, y, geoid = data["X"], data["y"], data["geoid"]

    pred, r2 = cross_val_ols(X, y)
    print(f"{X.shape[0]} places, {X.shape[1]} dimensions, {FOLDS}-fold out-of-fold R² = {r2:.3f}")

    out_path = output_path(DATA_DIR, OUT_STEM, ".parquet")
    pd.DataFrame({"geoid": geoid, "actual": y, "predicted": pred}).to_parquet(out_path)
    write_run_info(out_path, script=Path(__file__).name, source=acts_path.name, folds=FOLDS, seed=SEED)
    print(f"predictions -> {out_path}")


if __name__ == "__main__":
    main()
