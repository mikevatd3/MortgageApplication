"""Direct logit attribution for Gemma 2 in TransformerLens: split a token's logit into per-layer contributions."""
import torch

from mortgageapplication import continuation_joiner


def first_token_id(model, prompt: str, text: str) -> int:
    """Id of the first token the model would emit to start `text` right after the prompt."""
    prompt_ids = model.to_tokens(prompt)[0]
    full_ids = model.to_tokens(f"{prompt}{continuation_joiner(prompt)}{text}")[0]
    assert torch.equal(full_ids[:len(prompt_ids)], prompt_ids), "appending the text changed the prompt tokens"
    return full_ids[len(prompt_ids)].item()


def undo_softcap(logits, model):
    """Gemma 2 squashes its final logits with tanh; the residual stream decomposes the values before that."""
    cap = model.cfg.output_logits_soft_cap
    return torch.atanh(logits / cap) * cap if cap > 0 else logits


def residual_contributions(cache, n_layers):
    """What the embedding, then each layer's attention and MLP, actually added to the last position's stream.

    Differences of the residual stream at each stage, rather than the block output hooks: Gemma 2 applies
    extra norms and scaling around the blocks, so the hooked outputs don't add up to the stream, but
    these differences do by construction.
    """
    labels = ["embed"]
    parts = [cache["blocks.0.hook_resid_pre"][0, -1]]
    for layer in range(n_layers):
        pre, mid, post = (cache[f"blocks.{layer}.{hook}"][0, -1]
                          for hook in ("hook_resid_pre", "hook_resid_mid", "hook_resid_post"))
        labels += [f"{layer}_attn_out", f"{layer}_mlp_out"]
        parts += [mid - pre, post - mid]
    return torch.stack(parts), labels


def token_dla(cache, logits, model, token_ids):
    """Each component's contribution to the logit of each token at the last position.

    Returns (labels, dla, logits, gap): `dla` is [components, tokens]; `logits` are the real logits before
    Gemma's softcap; `gap` is how far each token's components (plus the unembed bias) are from its logit.
    Any contrast between tokens is a subtraction of columns of `dla`.
    """
    parts, labels = residual_contributions(cache, model.cfg.n_layers)
    parts = parts / cache["ln_final.hook_scale"][0, -1]   # the final norm's scale at the last position
    dla = parts @ model.W_U[:, token_ids]

    raw = undo_softcap(torch.as_tensor(logits)[0, -1], model)[token_ids]
    gap = dla.sum(0) + model.b_U[token_ids] - raw
    return labels, dla, raw, gap
