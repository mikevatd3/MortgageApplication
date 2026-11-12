import numpy as np
import pandas as pd
import torch


def summarize_strings(strings, model, tok, hook_name, skip_first=True):
    """
    Build a table with one row per string: the string, its token length, its
    tokens, and the residual activation at `hook_name` averaged over token
    positions.

    The model always runs the usual way, with the BOS token it prepends.
    skip_first=True drops that first position from everything reported:
    n_tokens and tokens cover only the string itself, and the average skips it.
    skip_first=False keeps it: n_tokens counts the BOS, tokens starts with it,
    and it is included in the average. `tok` is only used to turn token ids
    into readable token strings.

    Columns: string, n_tokens, tokens (list[str]), activation (np.ndarray of
    shape (d_model,); all NaN if no positions are left to average).
    """
    start = 1 if skip_first else 0

    rows = []
    with torch.no_grad():
        for s in strings:
            all_ids = model.to_tokens(s)
            ids = all_ids[:, start:]
            _, cache = model.run_with_cache(all_ids, names_filter=hook_name)
            resid = cache[hook_name][0, start:]

            if resid.shape[0]:
                activation = resid.mean(0).float().cpu().numpy()
            else:
                activation = np.full(resid.shape[-1], np.nan)

            rows.append({
                "string": s,
                "n_tokens": ids.shape[1],
                "tokens": tok.convert_ids_to_tokens(ids[0].tolist()),
                "activation": activation,
            })

    return pd.DataFrame(rows)
