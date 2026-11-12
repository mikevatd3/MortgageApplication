from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import tqdm

from mortgageapplication import load_prompt_template

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "google/gemma-2-9b"
OUT_PATH = DATA_DIR / "approvals__completion.csv"

# The pair of decision words whose first-token logits are recorded.
WORDS = ("approve", "review")

CONST_COLS = ["credit_score", "property_value", "loan_amount", "income", "debt_to_income"]
VARIANT_COLS = {
    "full": ["last_name", "first_name", "city", "state"] + CONST_COLS,
    "name_only": ["last_name", "first_name"] + CONST_COLS,
    "city_only": ["city", "state"] + CONST_COLS,
}


def first_token_id(tok, prompt: str, word: str) -> int:
    """Id of the first token the model would emit to start `word` right after the prompt."""
    prompt_ids = tok(prompt).input_ids
    word_ids = tok(f"{prompt} {word}").input_ids
    assert word_ids[:len(prompt_ids)] == prompt_ids, "appending the word changed the prompt tokens"
    return word_ids[len(prompt_ids)]


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
    model.eval()

    prompt_template = load_prompt_template("loan_approval_completion.txt")

    rows = []
    for _, file in tqdm.tqdm(manifest.iterrows(), total=len(manifest)):
        frame = pd.read_csv(DATA_DIR / f"{file.fileid}.csv")

        # One applicant per prompt: the test applicant, at row `position` of the table.
        for variant, cols in VARIANT_COLS.items():
            application = frame.loc[[int(file.position)], cols].to_csv(index=False)
            prompt = prompt_template.format(application=application)
            word_ids = [first_token_id(tok, prompt, w) for w in WORDS]

            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1].float()
            logprobs = torch.log_softmax(logits, dim=-1)

            row = {
                "batch": file.batch,
                "position": file.position,
                "race": file.race,
                "fileid": file.fileid,
                "variant": variant,
                "top_token": tok.decode(logits.argmax().item()),
            }
            for word, token_id in zip(WORDS, word_ids):
                row[f"logit_{word}"] = logits[token_id].item()
                row[f"logprob_{word}"] = logprobs[token_id].item()
            rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_PATH, index=False)
    print(f"{len(results)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
