"""
Draft measurement strategy: how does gemma-2-9b-it order a list of dates?

1. Behaviour. Greedy-generate the answer and check it against the true order.
2. Slot readout. Teacher-force the model's answer. At each position that is
   about to emit a date, read the next-token distribution over the first token
   of each input date (the month). This gives a per-slot "which date wins"
   measurement, and the logit-diff of the emitted date vs. the runner-up.
3. Direct logit attribution. For each slot, split the logit diff into
   per-head and per-MLP contributions to see which components push the
   selected date.
4. Attention. For the top heads from (3), look at where the slot position
   attends in the input list, and test whether it tracks date rank.
5. Causal check (TODO). Patch / ablate the top components, or swap two input
   dates, and see whether the emitted order changes the way the story predicts.
"""

import time
from dataclasses import dataclass

import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

load_dotenv()

MODEL = "google/gemma-2-9b-it"
MONTHS = [
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
]  # fmt: skip

PROMPT = """Please put the following dates in order:

August 2nd, March 15, December 21, October 31, June 21, July 20"""

TOP_K_HEADS = 10


@dataclass
class Slot:
    """One place in the response where the model chooses a date."""

    index: int  # 0-based place in the sorted list
    pos: int  # position whose output predicts the date's first token
    emitted: str  # month the model actually wrote


def month_of(token: str) -> str | None:
    word = token.strip(" *#.\n")
    return word if word in MONTHS else None


def build_tokens(model: HookedTransformer) -> tuple[torch.Tensor, int]:
    """Chat-formatted prompt tokens and the length of the prompt."""
    chat = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        add_generation_prompt=True,
        tokenize=False,
    )
    tokens = model.to_tokens(chat, prepend_bos=False)  # template adds <bos>
    return tokens, tokens.shape[1]


def generate_answer(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Step 1: greedy decode. Returns prompt + response tokens."""
    return model.generate(
        tokens,
        max_new_tokens=100,
        do_sample=False,
        stop_at_eos=True,
        eos_token_id=model.tokenizer.convert_tokens_to_ids("<end_of_turn>"),
        return_type="tokens",
        prepend_bos=False,
    )


def find_slots(model: HookedTransformer, full: torch.Tensor, prompt_len: int) -> list[Slot]:
    """Positions in the response that predict the month of each listed date."""
    slots = []
    for pos in range(prompt_len, full.shape[1]):
        month = month_of(model.tokenizer.decode(full[0, pos]))
        if month is not None:
            slots.append(Slot(len(slots), pos - 1, month))
    return slots


def candidate_ids(model: HookedTransformer) -> dict[str, int]:
    """Token id for each month name as it appears in the response (no space,
    after '**' or '. ')."""
    ids = {}
    for m in MONTHS:
        toks = model.to_tokens(m, prepend_bos=False)[0]
        ids[m] = toks[0].item()
    return ids


def slot_readout(model, logits, slots, cand: dict[str, int], listed: list[str]):
    """Step 2: probability of each listed month at each slot."""
    print(f"{'slot':>4} {'emitted':10} " + " ".join(f"{m[:3]:>6}" for m in listed))
    for s in slots:
        probs = logits[0, s.pos].softmax(-1)
        row = " ".join(f"{probs[cand[m]].item():6.3f}" for m in listed)
        print(f"{s.index + 1:>4} {s.emitted:10} {row}")


def direct_logit_attribution(model, cache, slots, cand, listed):
    """Step 3: per-component contribution to (emitted - best other listed date).

    Returns {slot index: (labels, contributions)}. Needs
    model.cfg.use_attn_result = True before the forward pass.
    """
    results = {}
    for s in slots:
        others = [cand[m] for m in listed if m != s.emitted]
        w_emit = model.W_U[:, cand[s.emitted]]
        # runner-up = strongest listed competitor under the model's own logits
        final = cache["ln_final.hook_normalized"][0, s.pos] @ model.W_U
        runner = max(others, key=lambda t: final[t].item())
        direction = w_emit - model.W_U[:, runner]

        # head results + MLPs + embed, scaled by the final layer norm
        parts, labels = cache.get_full_resid_decomposition(
            layer=-1, pos_slice=s.pos, apply_ln=True, return_labels=True,
            expand_neurons=False,
        )
        results[s.index] = (labels, (parts[:, 0] @ direction).detach().cpu())
    return results


def report_top_components(results):
    for idx, (labels, contrib) in results.items():
        top = contrib.abs().topk(TOP_K_HEADS).indices
        print(f"\nslot {idx + 1}: top components by |logit diff contribution|")
        for i in top:
            print(f"  {labels[i]:12s} {contrib[i].item():+.2f}")


def attention_to_dates(model, cache, slot, layer, head, prompt_tokens, prompt_len):
    """Step 4: where does (layer, head) look from this slot, within the prompt?"""
    pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head, slot.pos, :prompt_len]
    strs = model.to_str_tokens(prompt_tokens[0])
    for i in pattern.topk(5).indices.sort().values:
        print(f"  L{layer}H{head} slot {slot.index + 1} -> {strs[i]!r:12} {pattern[i].item():.3f}")


# TODO step 5: causal checks.
#   - swap two dates in the prompt (clean vs. corrupted), patch resid_pre /
#     head outputs from clean into corrupted at the slot positions, and track
#     the logit diff recovered per component (transformer_lens.patching).
#   - zero-ablate the top heads from step 3 and see whether the output order
#     degrades, and which slots break first.


def main():
    model = HookedTransformer.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    print(f"loaded on {model.cfg.device}, {model.cfg.n_layers} layers", flush=True)

    tokens, prompt_len = build_tokens(model)
    print(f"prompt is {prompt_len} tokens; generating...", flush=True)
    t0 = time.time()
    full = generate_answer(model, tokens)
    print(f"generated {full.shape[1] - prompt_len} tokens in {time.time() - t0:.0f}s", flush=True)
    print(model.tokenizer.decode(full[0, prompt_len:]), flush=True)

    model.set_use_attn_result(True)  # per-head outputs for DLA, only needed for the cached pass

    slots = find_slots(model, full, prompt_len)
    cand = candidate_ids(model)
    listed = ["August", "March", "December", "October", "June", "July"]
    print("emitted order:", [s.emitted for s in slots])
    print("correct order:", sorted(listed, key=MONTHS.index))

    print("running cached forward pass...", flush=True)
    with torch.no_grad():
        logits, cache = model.run_with_cache(full, prepend_bos=False)

    slot_readout(model, logits, slots, cand, listed)

    results = direct_logit_attribution(model, cache, slots, cand, listed)
    report_top_components(results)

    # attention for the strongest head in each slot
    for s in slots:
        labels, contrib = results[s.index]
        head_idx = [i for i, l in enumerate(labels) if l.startswith("L") and "H" in l]
        best = max(head_idx, key=lambda i: abs(contrib[i].item()))
        layer, head = (int(x) for x in labels[best][1:].split("H"))
        attention_to_dates(model, cache, s, layer, head, tokens, prompt_len)


if __name__ == "__main__":
    main()
