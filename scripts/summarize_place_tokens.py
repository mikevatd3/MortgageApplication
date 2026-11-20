from pathlib import Path
import io
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer

from mortgageapplication.outputs import output_path, write_run_info
from mortgageapplication.token_table import summarize_strings

load_dotenv()

MODEL = "google/gemma-2-9b"
# mid-depth of Gemma-2-9b's 42 layers, where semantic features tend to peak (and Gemma Scope has a residual SAE)
HOOK = "blocks.20.hook_resid_post"
DATA_DIR = Path(__file__).parent.parent / "data"
PLACES_PATH = DATA_DIR / "places_20261112.parquet"
OUT_STEM = "place_city_token_summaries"

FLOAT16_MAX = float(np.finfo(np.float16).max)


def to_float16(vector):
    """Cast an activation vector to float16, refusing to silently overflow to inf."""
    if np.abs(vector).max() > FLOAT16_MAX:
        raise ValueError(f"activation value outside float16 range (max {FLOAT16_MAX:.0f})")
    return vector.astype(np.float16)


def check_float16_parquet():
    """Fail now, not after loading the model, if parquet can't round-trip float16 lists."""
    buffer = io.BytesIO()
    pd.DataFrame({"activation": [np.zeros(4, dtype=np.float16)]}).to_parquet(buffer)
    buffer.seek(0)
    pd.read_parquet(buffer)


def main():
    check_float16_parquet()

    places = pd.read_parquet(PLACES_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device)
    model.eval()

    table = summarize_strings(places["city"].tolist(), model, model.tokenizer, HOOK)
    table["activation"] = table["activation"].map(to_float16)

    # rows come back in input order; geoid keeps same-named cities apart and joins to the race shares
    table.insert(0, "geoid", places["geoid"].tolist())

    out_path = output_path(DATA_DIR, OUT_STEM, ".parquet")
    table.to_parquet(out_path)
    write_run_info(out_path, script=Path(__file__).name, model=MODEL, hook=HOOK)
    print(f"{len(table)} places -> {out_path}")


if __name__ == "__main__":
    main()
