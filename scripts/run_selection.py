from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import tqdm

from mortgageapplication import load_prompt_template

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "google/gemma-2-9b"
OUT_PATH = DATA_DIR / "selections__completion.csv"


def validate_selection(output: str, frame: pd.DataFrame) -> tuple[int | None, str | None]:
    """Return (row position, None) if output names exactly one applicant in the table, else (None, reason)."""
    text = " ".join(output.upper().split())
    full_names = (frame.first_name + " " + frame.last_name).str.upper().str.split().str.join(" ")

    matches = full_names.index[full_names == text].tolist()
    if not matches:
        return None, "not the name of an applicant in the table"
    if len(matches) > 1:
        return None, "name matches more than one applicant"

    return matches[0], None


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")

    prompt_template = load_prompt_template("loan_selection_completion.txt")

    rows = []
    for _, file in tqdm.tqdm(manifest.iterrows(), total=len(manifest)):
        frame = pd.read_csv(DATA_DIR / f"{file.fileid}.csv")

        prompt = prompt_template.format(table=frame.to_csv(index=False), n=len(frame))

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False)

        new_tokens = out[0][inputs.input_ids.shape[1]:]
        output = tok.decode(new_tokens, skip_special_tokens=True).strip().split("\n")[0]
        if not output.strip():
            tqdm.tqdm.write(f"{file.fileid}: empty output, raw={tok.decode(new_tokens)!r}")

        selected, error = validate_selection(output, frame)
        if error:
            tqdm.tqdm.write(f"{file.fileid}: invalid selection ({error}): {output.strip()!r}")

        rows.append({
            "batch": file.batch,
            "position": file.position,
            "race": file.race,
            "fileid": file.fileid,
            "selection": output.strip(),
            "selected_row": selected,
            "valid": selected is not None,
            "error": error,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False)
    print(f"{results.valid.sum()}/{len(results)} valid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
