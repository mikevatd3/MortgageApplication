from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication import load_prompt_template

load_dotenv()

MODEL = "google/gemma-2-9b"
# mid-depth of Gemma-2-9b's 42 layers, where semantic features tend to peak (and Gemma Scope has a residual SAE)
HOOK = "blocks.20.hook_resid_post"
DATA_DIR = Path(__file__).parent.parent / "data"
PLACES_PATH = DATA_DIR / "places_20261112.parquet"
OUT_PATH = DATA_DIR / "place_activations_gemma-2-9b.npz"


def to_prompt(template, city, state):
    return template.format(city=city.title(), state=state.title())


def main():
    template = load_prompt_template("simple_place.txt")
    frame = pd.read_parquet(PLACES_PATH)
    prompts = [to_prompt(template, c, s) for c, s in zip(frame.city, frame.state)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype="bfloat16")
    model.eval()

    acts = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            _, cache = model.run_with_cache(prompt, names_filter=HOOK)
            acts.append(cache[HOOK][0, -1].float().cpu().numpy())

    # geoid rides along so predictions can be joined back to the places
    np.savez(OUT_PATH, X=np.stack(acts), y=frame.bla.values, geoid=np.array(frame.geoid.tolist(), dtype=str))


if __name__ == "__main__":
    main()
