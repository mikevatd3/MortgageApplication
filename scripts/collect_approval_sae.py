from pathlib import Path
import pandas as pd
import torch
import tqdm
from sae_lens import SAE
from transformer_lens import HookedTransformer

from mortgageapplication import fill_application_prompt, load_prompt_template
from mortgageapplication.dla import first_token_id
from mortgageapplication.outputs import output_path, write_run_info


MODEL = "gemma-2-9b"
# Gemma Scope residual SAE at layer 20 (mid-depth of Gemma-2-9b's 42 layers), 16k features
SAE_RELEASE = "gemma-scope-9b-pt-res-canonical"
SAE_ID = "layer_20/width_16k/canonical"
HOOK = "blocks.20.hook_resid_post"

DATA_DIR = Path(__file__).parent.parent / "data"
LISTINGS_PATH = DATA_DIR / "boundary_listings_2026-11-20.csv"
PROMPT = "loan_approval_completion.txt"
FEATURES_STEM = "approval_sae_features_l20_16k"
PROMPTS_STEM = "approval_sae_prompts_l20_16k"

WORDS = ("approve", "review")
# The error term (what the SAE fails to reconstruct) is stored as a pseudo-feature with this id.
ERROR = -1


def span_positions(tokenizer, token_ids, prompt, anchor, text):
    """Token positions (the BOS counts as 0) covered by `text`, which follows `anchor` in the prompt."""
    start = prompt.index(anchor) + len(anchor)
    end = start + len(text)
    enc = tokenizer(prompt, return_offsets_mapping=True)
    assert enc["input_ids"] == token_ids, "the tokenizer's ids differ from the model's"
    positions = [i for i, (a, b) in enumerate(enc["offset_mapping"]) if a < end and b > start]
    assert positions, f"no tokens found for {text!r}"
    return positions


def attribute(model, sae, prompt, approve_id, review_id):
    """Run the prompt with the SAE spliced in at HOOK.

    Returns the SAE's feature activations and reconstruction error at every position, the gradient of the
    approve-minus-review logit difference with respect to each, and the logit difference. The spliced value
    equals the original residual exactly (reconstruction plus error), so the output is the model's own.
    """
    store = {}

    def splice(resid, hook):
        feats = sae.encode(resid).detach().requires_grad_(True)
        recon = sae.decode(feats)
        error = (resid - recon).detach().requires_grad_(True)
        store.update(feats=feats, error=error, resid=resid.detach(), recon=recon.detach())
        return recon + error

    with torch.enable_grad():
        logits = model.run_with_hooks(prompt, fwd_hooks=[(HOOK, splice)])
        logit_diff = logits[0, -1, approve_id] - logits[0, -1, review_id]
        grad_feats, grad_error = torch.autograd.grad(logit_diff, [store["feats"], store["error"]])

    return store, grad_feats[0], grad_error[0], logit_diff.item()


def main():
    listings = pd.read_csv(LISTINGS_PATH)
    listings["name"] = listings["name"].str.title()
    template = load_prompt_template(PROMPT)

    model = HookedTransformer.from_pretrained(MODEL)
    model.eval()
    model.requires_grad_(False)

    sae = SAE.from_pretrained(SAE_RELEASE, SAE_ID, device=str(model.cfg.device))
    sae.requires_grad_(False)
    assert sae.cfg.metadata.hook_name == HOOK, f"the SAE reads {sae.cfg.metadata.hook_name}, not {HOOK}"

    feature_rows, prompt_rows = [], []
    for prompt_id, listing in tqdm.tqdm(listings.iterrows(), total=len(listings)):
        prompt = fill_application_prompt(template, listing)
        approve_id, review_id = (first_token_id(model, prompt, w) for w in WORDS)

        tokens = model.to_tokens(prompt)[0]
        str_tokens = model.tokenizer.convert_ids_to_tokens(tokens.tolist())
        positions = {
            "name": span_positions(model.tokenizer, tokens.tolist(), prompt, "NAME: ", listing["name"]),
            "city": span_positions(model.tokenizer, tokens.tolist(), prompt, "CITY: ", listing["city"]),
            "last": [len(tokens) - 1],
        }

        store, grad_feats, grad_error, logit_diff = attribute(model, sae, prompt, approve_id, review_id)
        feats, error, resid = store["feats"][0], store["error"][0], store["resid"][0]

        prompt_row = {
            "prompt_id": prompt_id, "set": listing["set"], "top": listing["top"],
            "name": listing["name"], "city": listing["city"], "p_denial": listing["p_denial"],
            "logit_diff": logit_diff,
        }
        for role, pos in positions.items():
            f, g = feats[pos], grad_feats[pos]
            active = (f > 0).any(0).nonzero().squeeze(1)

            # Summed over the span's tokens, so names that split into different numbers of tokens still compare.
            act = f.sum(0)[active]
            attr = (f * g).sum(0)[active]
            ids = active.tolist() + [ERROR]
            acts = act.tolist() + [error[pos].norm(dim=-1).sum().item()]
            attrs = attr.tolist() + [(error[pos] * grad_error[pos]).sum().item()]
            feature_rows.append(pd.DataFrame({
                "prompt_id": prompt_id, "role": role, "feature": ids, "act": acts, "attr": attrs,
            }))

            prompt_row[f"{role}_recon_error"] = (error[pos].pow(2).sum() / resid[pos].pow(2).sum()).item()
            prompt_row[f"{role}_l0"] = (f > 0).sum(-1).float().mean().item()
            prompt_row[f"{role}_tokens"] = "|".join(str_tokens[p] for p in pos)
        prompt_rows.append(prompt_row)

    features = pd.concat(feature_rows, ignore_index=True)
    features = features.astype({"prompt_id": "int32", "feature": "int32", "act": "float32", "attr": "float32"})
    prompts = pd.DataFrame(prompt_rows)

    info = dict(script=Path(__file__).name, prompts=[PROMPT], model=MODEL, sae_release=SAE_RELEASE,
                sae_id=SAE_ID, listings=LISTINGS_PATH.name)
    for stem, table in ((FEATURES_STEM, features), (PROMPTS_STEM, prompts)):
        out_path = output_path(DATA_DIR, stem, ".parquet")
        table.to_parquet(out_path)
        write_run_info(out_path, **info)
        print(f"{len(table)} rows -> {out_path}")


if __name__ == "__main__":
    main()
