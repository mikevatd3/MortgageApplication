from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2")
tokens = model.to_tokens("John and Mary went to the store. John gave a drink to")
_, cache = model.run_with_cache(tokens)

resid, labels = cache.accumulated_resid(incl_mid=True, pos_slice=-1, return_labels=True)
probs = model.unembed(model.ln_final(resid)).softmax(-1)[:, 0]  # [component, vocab]

target = model.to_single_token(" Mary")
for label, p in zip(labels, probs):
    print(f"{label:12s} top={model.tokenizer.decode(p.argmax())!r:12} p(Mary)={p[target]:.3f}")
