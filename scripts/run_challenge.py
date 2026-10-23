from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import tqdm

from mortgageapplication import load_prompt_template

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "google/gemma-2-9b-it"
OUT_PATH = DATA_DIR / "rankings__name_only.csv"

INT_LIST = re.compile(r"\d+(,\d+)*")

CONST_COLS = ["credit_score", "property_value", "loan_amount", "income", "debt_to_income"]
CITY_ONLY_COLS = ["city", "state"] + CONST_COLS
NAME_ONLY_COLS = ["first_name", "last_name"] + CONST_COLS


def validate_ranking(output: str, n: int) -> tuple[list[int] | None, str | None]:
    """Return (ranking, None) if output is a comma-separated permutation of 0..n-1, else (None, reason)."""
    text = output.strip()

    if not INT_LIST.fullmatch(text):
        return None, "not a comma-separated list of integers"
    ranking = [int(x) for x in text.split(",")]

    if sorted(ranking) != list(range(n)):
        return None, f"not each id in 0..{n - 1} exactly once"

    return ranking, None


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    prompt_template = load_prompt_template("loan_prioritization.txt")

    rows = []
    for _, file in tqdm.tqdm(manifest.iterrows(), total=len(manifest)):
        frame = pd.read_csv(DATA_DIR / f"{file.fileid}.csv")
        frame = frame[NAME_ONLY_COLS]

        prompt = prompt_template.format(table=frame.to_csv(), n=len(frame))

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        new_tokens = out[0][inputs.input_ids.shape[1]:]
        output = tok.decode(new_tokens, skip_special_tokens=True)
        if not output.strip():
            tqdm.tqdm.write(f"{file.fileid}: empty output, raw={tok.decode(new_tokens)!r}")

        ranking, error = validate_ranking(output, len(frame))
        rows.append({
            "batch": file.batch,
            "position": file.position,
            "race": file.race,
            "fileid": file.fileid,
            "ranking": output.strip(),
            "valid": ranking is not None,
            "error": error,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False)
    print(f"{results.valid.sum()}/{len(results)} valid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
