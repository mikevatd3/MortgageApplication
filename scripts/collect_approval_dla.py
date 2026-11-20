from pathlib import Path
import pandas as pd
import torch
import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication import fill_application_prompt, load_prompt_template
from mortgageapplication.dla import first_token_id, residual_contributions, undo_softcap
from mortgageapplication.outputs import output_path, write_run_info


MODEL = "gemma-2-9b"
DATA_DIR = Path(__file__).parent.parent / "data"
LISTINGS_PATH = DATA_DIR / "comparable_listings_2026-11-12_v3.csv"
OUT_STEM = "approval_dla_gemma-2-9b"

# Direct logit attribution is taken for the first token of the first word minus the first token of the second.
WORDS = ("approve", "review")

# How far the summed components may be from the real logit difference before a warning is printed.
TOLERANCE = 0.05


def main():
    listings = pd.read_csv(LISTINGS_PATH)
    listings["name"] = listings["name"].str.title() # Change for v2
    template = load_prompt_template("loan_approval_completion.txt")

    model = HookedTransformer.from_pretrained(MODEL)
    model.eval()

    rows = []
    for _, listing in tqdm.tqdm(listings.iterrows(), total=len(listings)):
        prompt = fill_application_prompt(template, listing)
        approve_id, review_id = (first_token_id(model, prompt, w) for w in WORDS)
        direction = model.W_U[:, approve_id] - model.W_U[:, review_id]

        with torch.no_grad():
            logits, cache = model.run_with_cache(prompt)

            comps, labels = residual_contributions(cache, model.cfg.n_layers)
            comps = comps / cache["ln_final.hook_scale"][0, -1]   # the final norm's scale at the last position
            dla = comps @ direction

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
                      f"  max |components.sum - final|: {(comps.sum(0) - final).abs().max().item():.4f}")

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
    out_path = output_path(DATA_DIR, OUT_STEM, ".parquet")
    results.to_parquet(out_path)
    write_run_info(out_path, script=Path(__file__).name, prompts=["loan_approval_completion.txt"],
                   model=MODEL, listings=LISTINGS_PATH.name)
    print(f"{len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
