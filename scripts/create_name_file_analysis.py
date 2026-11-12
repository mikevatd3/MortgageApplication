from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication.token_table import summarize_strings

MODEL = "gpt2-small"
HOOK = "blocks.6.hook_resid_post"
DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK = 1_000

NAME_FILES = {
    "first": DATA_DIR / "first_names.parquet",
    "last": DATA_DIR / "last_names.parquet",
}


def summarize_file(path, model):
    names = pd.read_parquet(path)["name"].tolist()

    chunks = [
        summarize_strings(names[i:i + CHUNK], model, model.tokenizer, HOOK)
        for i in tqdm(range(0, len(names), CHUNK), desc=path.stem)
    ]
    return pd.concat(chunks, ignore_index=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device)

    for kind, path in NAME_FILES.items():
        table = summarize_file(path, model)
        out_path = DATA_DIR / f"{kind}_name_token_summaries.parquet"
        table.to_parquet(out_path)
        print(f"{len(table)} {kind} names -> {out_path}")


if __name__ == "__main__":
    main()
