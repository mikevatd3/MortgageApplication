from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication import load_prompt_template
from mortgageapplication.outputs import output_path, write_run_info

load_dotenv()

MODEL = "google/gemma-2-9b"
# mid-depth of Gemma-2-9b's 42 layers, where semantic features tend to peak (and Gemma Scope has a residual SAE)
HOOK = "blocks.20.hook_resid_post"
DATA_DIR = Path(__file__).parent.parent / "data"
NAMES_PATH = DATA_DIR / "uniform_names_2026-10-24.csv"
OUT_STEM = "name_activations_gemma-2-9b"


def to_prompt(template, full_name):
    last, first = [p.strip().title() for p in full_name.split(",", 1)]
    return template.format(first=first, last=last)


def main():
    template = load_prompt_template("simple_name.txt")
    frame = pd.read_csv(NAMES_PATH)
    prompts = [to_prompt(template, n) for n in frame.full_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained(MODEL, device=device, dtype="bfloat16")
    model.eval()

    acts = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            _, cache = model.run_with_cache(prompt, names_filter=HOOK)
            acts.append(cache[HOOK][0, -1].float().cpu().numpy())

    # full_name rides along so predictions can be joined back to the names
    out_path = output_path(DATA_DIR, OUT_STEM, ".npz")
    np.savez(out_path, X=np.stack(acts), y=frame.bla_full.values, full_name=np.array(frame.full_name.tolist(), dtype=str))
    write_run_info(out_path, script=Path(__file__).name, prompts=["simple_name.txt"], model=MODEL, hook=HOOK)
    print(f"{len(acts)} names -> {out_path}")


if __name__ == "__main__":
    main()
