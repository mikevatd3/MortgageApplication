from pathlib import Path
import io
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication.token_table import summarize_strings

load_dotenv()

MODEL = "google/gemma-2-9b"
HOOK = "blocks.6.hook_resid_post"
DATA_DIR = Path(__file__).parent.parent / "data"
CHUNK = 1_000

NAME_FILES = {
    "first": DATA_DIR / "first_names.parquet",
    "last": DATA_DIR / "last_names.parquet",
}


FLOAT16_MAX = float(np.finfo(np.float16).max)


def to_float16(vector):
    """Cast an activation vector to float16, refusing to silently overflow to inf."""
    if np.abs(vector).max() > FLOAT16_MAX:
        raise ValueError(f"activation value outside float16 range (max {FLOAT16_MAX:.0f})")
    return vector.astype(np.float16)


def check_float16_parquet():
    """Fail now, not after hours of compute, if parquet can't round-trip float16 lists."""
    buffer = io.BytesIO()
    pd.DataFrame({"activation": [np.zeros(4, dtype=np.float16)]}).to_parquet(buffer)
    buffer.seek(0)
    pd.read_parquet(buffer)


def summarize_file(path, model):
    names = pd.read_parquet(path)["name"].tolist()

    chunks = []
    for i in tqdm(range(0, len(names), CHUNK), desc=path.stem):
        chunk = summarize_strings(names[i:i + CHUNK], model, model.tokenizer, HOOK)
        chunk["activation"] = chunk["activation"].map(to_float16)
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def main():
    check_float16_parquet()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device)
    model.eval()

    for kind, path in NAME_FILES.items():
        table = summarize_file(path, model)
        out_path = DATA_DIR / f"{kind}_name_token_summaries.parquet"
        table.to_parquet(out_path)
        print(f"{len(table)} {kind} names -> {out_path}")


if __name__ == "__main__":
    main()
