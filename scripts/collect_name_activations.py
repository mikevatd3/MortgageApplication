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
NAMES_PATH = DATA_DIR / "uniform_names_2026-10-24.csv"
OUT_PATH = DATA_DIR / "name_activations_gemma-2-9b.npz"


def to_prompt(template, full_name):
    last, first = [p.strip().title() for p in full_name.split(",", 1)]
    return template.format(first=first, last=last)


def main():
    template = load_prompt_template("simple_name.txt")
    frame = pd.read_csv(NAMES_PATH)
    prompts = [to_prompt(template, n) for n in frame.full_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype=torch.bfloat16)
    model.eval()

    acts = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            _, cache = model.run_with_cache(prompt, names_filter=HOOK)
            acts.append(cache[HOOK][0, -1].float().cpu().numpy())

    np.savez(OUT_PATH, X=np.stack(acts), y=frame.bla_full.values)


if __name__ == "__main__":
    main()
