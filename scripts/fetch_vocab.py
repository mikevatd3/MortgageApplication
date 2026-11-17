import pandas as pd
from transformers import AutoTokenizer

MODEL = "google/gemma-2-9b"


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    vocab = tokenizer.get_vocab()  # token string -> id
    special_ids = set(tokenizer.all_special_ids)

    frame = pd.DataFrame(
        {"token_id": list(vocab.values()), "token": list(vocab.keys())}
    ).sort_values("token_id", ignore_index=True)

    # Decoded form (e.g. "▁Hello" -> " Hello"), useful for matching against text
    frame["decoded"] = [tokenizer.decode([i]) for i in frame["token_id"]]
    frame["is_special"] = frame["token_id"].isin(special_ids)

    frame.to_parquet("data/vocab_gemma-2-9b.parquet", index=False)
    print(f"Wrote {len(frame)} tokens")


if __name__ == "__main__":
    main()
