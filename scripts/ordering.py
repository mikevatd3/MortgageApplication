""" 
Use direct logit attribution to see which components, heads and MLPs, push
the final logit for the predicted token. That tells you which heads matter.
Then, for the heads that matter, look at their attention patterns to see where
they’re pulling from. Then confirm causally with activation patching or
ablation, where you swap or zero a component and see if the prediction changes.
"""

from textwrap import dedent
from transformer_lens import HookedTransformer

def main():
    prompt = "New Year's day is the first day in"


    model = HookedTransformer.from_pretrained("gpt2")


    tokens = model.to_tokens(prompt)
    _, cache = model.run_with_cache(tokens)

    resid, labels = cache.accumulated_resid(incl_mid=True, pos_slice=-1, return_labels=True)
    probs = model.unembed(model.ln_final(resid)).softmax(-1)[:, 0]  # [component, vocab]

    target = model.to_single_token(" January")
    for label, p in zip(labels, probs):
        print(f"{label:12s} top={model.tokenizer.decode(p.argmax())!r:12} p(January)={p[target]:.3f}")

if __name__ == "__main__":
    main()
