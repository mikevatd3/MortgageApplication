import re

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

BLOCK_HOOK = re.compile(r"^blocks\.(\d+)\.")


def summarize_strings(strings, model, tok, hook_name, skip_first=True, batch_size=256):
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

    Strings run through the model `batch_size` at a time, right-padded. The
    model is causal, so padding never changes the activations at real
    positions, and only real positions are averaged. A `blocks.N.` hook also
    stops the forward pass after block N, since later layers can't affect it.

    Columns: string, n_tokens, tokens (list[str]), activation (np.ndarray of
    shape (d_model,); all NaN if no positions are left to average).
    """
    start = 1 if skip_first else 0

    block = BLOCK_HOOK.match(hook_name)
    stop_at_layer = int(block.group(1)) + 1 if block else None

    rows = []
    with torch.no_grad():
        for i in range(0, len(strings), batch_size):
            batch = strings[i:i + batch_size]
            ids = [model.to_tokens(s)[0] for s in batch]
            padded = pad_sequence(ids, batch_first=True, padding_value=0).to(model.cfg.device)

            _, cache = model.run_with_cache(
                padded, names_filter=hook_name, stop_at_layer=stop_at_layer,
            )
            resid = cache[hook_name]

            for s, row_ids, row_resid in zip(batch, ids, resid):
                real = row_resid[start:len(row_ids)]

                if real.shape[0]:
                    activation = real.mean(0).float().cpu().numpy()
                else:
                    activation = np.full(real.shape[-1], np.nan)

                rows.append({
                    "string": s,
                    "n_tokens": len(row_ids) - start,
                    "tokens": tok.convert_ids_to_tokens(row_ids[start:].tolist()),
                    "activation": activation,
                })

    return pd.DataFrame(rows)
