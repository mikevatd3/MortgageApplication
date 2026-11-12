from pathlib import Path
import pandas as pd
import torch
from transformer_lens import HookedTransformer

from mortgageapplication import load_prompt_template


MODEL = "gemma-2-9b"
DATA_DIR = Path(__file__).parent.parent / "data"
LISTINGS_PATH = DATA_DIR / "comparable_listings_2026-11-12.csv"
OUT_PATH = DATA_DIR / "approval_dla_gemma-2-9b.parquet"

# Direct logit attribution is taken for the first token of the first word minus the first token of the second.
WORDS = ("approve", "review")

# How far the summed components may be from the real logit difference before a warning is printed.
TOLERANCE = 0.05


def first_token_id(model, prompt: str, word: str) -> int:
    """Id of the first token the model would emit to start `word` right after the prompt."""
    prompt_ids = model.to_tokens(prompt)[0]
    word_ids = model.to_tokens(f"{prompt} {word}")[0]
    assert torch.equal(word_ids[:len(prompt_ids)], prompt_ids), "appending the word changed the prompt tokens"
    return word_ids[len(prompt_ids)].item()


def undo_softcap(logits, model):
    """Gemma 2 squashes its final logits with tanh; the residual stream decomposes the values before that."""
    cap = model.cfg.output_logits_soft_cap
    return torch.atanh(logits / cap) * cap if cap > 0 else logits


def main():
    listings = pd.read_csv(LISTINGS_PATH)
    template = load_prompt_template("loan_approval_completion.txt")

    model = HookedTransformer.from_pretrained(MODEL)
    model.eval()

    rows = []
    for _, listing in listings.iterrows():
        prompt = template.format(**listing)
        approve_id, review_id = (first_token_id(model, prompt, w) for w in WORDS)
        direction = model.W_U[:, approve_id] - model.W_U[:, review_id]

        with torch.no_grad():
            logits, cache = model.run_with_cache(prompt)

            # Per-layer attention and MLP outputs, not per-head: Gemma 2 normalizes each block's output
            # after the heads are summed, so per-head contributions don't add up to the residual stream.
            comps, labels = cache.get_full_resid_decomposition(
                layer=-1, expand_neurons=False, apply_ln=True,
                pos_slice=-1, return_labels=True)
            dla = (comps @ direction).squeeze()

            # Sanity: the components plus the unembed bias should rebuild the logit difference.
            bias_term = model.b_U[approve_id] - model.b_U[review_id]
            raw = undo_softcap(torch.as_tensor(logits)[0, -1], model)
            logit_diff = raw[approve_id] - raw[review_id]
            gap = (dla.sum() + bias_term - logit_diff).item()
            if abs(gap) > TOLERANCE:
                # Localize the gap: do the components rebuild the final normalized residual, and does that
                # residual reproduce the logit difference?
                final = cache["ln_final.hook_normalized"][0, -1]
                capped = torch.as_tensor(logits)[0, -1]
                print(f"warning: {listing['name']}: components sum to {(dla.sum() + bias_term).item():.3f}, "
                      f"logit difference is {logit_diff.item():.3f} (gap {gap:+.3f})\n"
                      f"  final residual @ direction:   {(final @ direction + bias_term).item():.3f}\n"
                      f"  capped logit difference:      {(capped[approve_id] - capped[review_id]).item():.3f} "
                      f"(softcap {model.cfg.output_logits_soft_cap})\n"
                      f"  max |components.sum - final|: {(comps.sum(0).squeeze() - final).abs().max().item():.4f}")

        for label, value in zip(labels, dla.float().cpu().numpy()):
            rows.append({
                "set": listing["set"],
                "top": listing["top"],
                "name": listing["name"],
                "component": label,
                "dla": float(value),
                "logit_diff": logit_diff.item(),
                "decomposition_gap": gap,
            })

    results = pd.DataFrame(rows)
    results.to_parquet(OUT_PATH)
    print(f"{len(results)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
