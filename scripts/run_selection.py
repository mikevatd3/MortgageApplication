from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import pandas as pd
import re
import tqdm

from mortgageapplication import load_prompt_template

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "google/gemma-2-9b"
OUT_PATH = DATA_DIR / "selections__completion.csv"


def clean_selection(output: str) -> str:
    """Normalize "FIRST,LAST", "FIRST, LAST" and repeated spaces to "FIRST LAST"."""
    return " ".join(re.sub(r"\s*,\s*", " ", output.upper()).split())


def first_token_ids(tok, prompt: str, frame: pd.DataFrame) -> list[int]:
    """Id of the first token the model would emit to start each applicant's name, in row order."""
    prompt_ids = tok(prompt).input_ids
    ids = []
    for first, last in zip(frame.first_name, frame.last_name):
        name_ids = tok(f"{prompt} {first} {last}").input_ids
        assert name_ids[:len(prompt_ids)] == prompt_ids, "name tokenization changed the prompt tokens"
        ids.append(name_ids[len(prompt_ids)])
    return ids


def validate_selection(output: str, frame: pd.DataFrame) -> tuple[int | None, str | None]:
    """Return (row position, None) if output names exactly one applicant in the table, else (None, reason)."""
    text = clean_selection(output)
    full_names = (frame.first_name + " " + frame.last_name).map(clean_selection)

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

    prompt_template = load_prompt_template("loan_selection_criteria_completion.txt")

    rows = []
    for _, file in tqdm.tqdm(manifest.iterrows(), total=len(manifest)):
        frame = pd.read_csv(DATA_DIR / f"{file.fileid}.csv")

        prompt = prompt_template.format(table=frame.to_csv(index=False), n=len(frame))

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=20, do_sample=False,
            return_dict_in_generate=True, output_logits=True,
        )

        # out.logits[0] is the raw next-token distribution at the position where the name starts.
        start_ids = first_token_ids(tok, prompt, frame)
        start_logits = out.logits[0][0, start_ids].float().tolist()

        new_tokens = out.sequences[0][inputs.input_ids.shape[1]:]
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
            "selection_clean": clean_selection(output),
            "start_tokens": json.dumps([tok.decode(i) for i in start_ids]),
            "start_logits": json.dumps(start_logits),
            "selected_row": selected,
            "valid": selected is not None,
            "error": error,
        })

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False)
    print(f"{results.valid.sum()}/{len(results)} valid -> {OUT_PATH}")


if __name__ == "__main__":
    main()
