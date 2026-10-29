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
HOOK = "ln_final.hook_normalized"
DATA_DIR = Path(__file__).parent.parent / "data"
PLACES_PATH = DATA_DIR / "places_2026-10-29.csv"
OUT_PATH = DATA_DIR / "place_activations_gemma-2-9b.npz"


def to_prompt(template, city, state):
    return template.format(city=city.title(), state=state.title())


def main():
    template = load_prompt_template("simple_place.txt")
    frame = pd.read_csv(PLACES_PATH)
    prompts = [to_prompt(template, c, s) for c, s in zip(frame.city, frame.state)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.bfloat16)
    model.eval()

    acts = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            _, cache = model.run_with_cache(prompt, names_filter=HOOK)
            acts.append(cache[HOOK][0, -1].float().cpu().numpy())

    np.savez(OUT_PATH, X=np.stack(acts), y=frame.bla.values)


if __name__ == "__main__":
    main()
