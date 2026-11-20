from pathlib import Path
import pandas as pd
import torch
import tqdm
from transformer_lens import HookedTransformer

from mortgageapplication import load_prompt_template
from mortgageapplication.applicants import last_name_first, title_case_names
from mortgageapplication.dla import first_token_id, token_dla
from mortgageapplication.outputs import output_path, write_run_info

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "gemma-2-9b"
PROMPT = "loan_selection_criteria_completion.txt"
OUT_STEM = "selection_dla_gemma-2-9b"

# How far a token's summed components may be from its real logit before a warning is printed.
TOLERANCE = 0.05
# Warn when less of the next-token probability than this lands on the applicants' first tokens: the measured
# tokens probably aren't the ones the model would actually emit.
MIN_CANDIDATE_MASS = 0.5


def name_text(last: str, first: str) -> str:
    return f"{last}, {first}"


def main():
    manifest = pd.read_csv(DATA_DIR / "manifest.csv")
    template = load_prompt_template(PROMPT)

    # The baseline is the white version of each batch's test applicant, scored in every race's prompt.
    white = manifest[manifest.race == "whi"].set_index("batch")
    white_name = {
        batch: name_text(row.last_name.title(), row.first_name.title()) for batch, row in white.iterrows()
    }

    model = HookedTransformer.from_pretrained(MODEL)
    model.eval()

    rows = []
    for _, file in tqdm.tqdm(manifest.iterrows(), total=len(manifest)):
        frame = last_name_first(title_case_names(pd.read_csv(DATA_DIR / f"{file.fileid}.csv")))
        prompt = template.format(table=frame.to_csv(index=False), n=len(frame))
        position = int(file.position)

        names = [name_text(last, first) for last, first in zip(frame.last_name, frame.first_name)]
        candidate_ids = [first_token_id(model, prompt, name) for name in names]
        baseline_id = first_token_id(model, prompt, white_name[file.batch])
        token_ids = candidate_ids + [baseline_id]

        with torch.no_grad():
            logits, cache = model.run_with_cache(prompt)
            labels, dla, raw, gap = token_dla(cache, logits, model, token_ids)

            # The real next-token distribution is the capped one.
            capped = torch.as_tensor(logits)[0, -1].float()
            probs = torch.softmax(capped, dim=-1)
            candidate_mass = probs[sorted(set(candidate_ids))].sum().item()
            top_token = model.tokenizer.decode(capped.argmax().item())

        if gap.abs().max().item() > TOLERANCE:
            tqdm.tqdm.write(f"{file.fileid}: components miss a logit by up to {gap.abs().max().item():.3f}")
        if candidate_mass < MIN_CANDIDATE_MASS:
            tqdm.tqdm.write(f"{file.fileid}: only {candidate_mass:.2f} of the probability is on the applicants' "
                            f"first tokens (top token {top_token!r})")

        collision = len(set(candidate_ids)) < len(candidate_ids)
        dla = dla.float().cpu().numpy()
        raw = raw.float().cpu().numpy()
        gap = gap.float().cpu().numpy()

        for j, token_id in enumerate(token_ids):
            is_baseline = j == len(candidate_ids)
            role = "white_baseline" if is_baseline else ("test" if j == position else "rival")
            for i, label in enumerate(labels):
                rows.append({
                    "batch": file.batch,
                    "position": position,
                    "race": file.race,
                    "fileid": file.fileid,
                    "role": role,
                    "row": -1 if is_baseline else j,
                    "token_id": token_id,
                    "token": model.tokenizer.decode(token_id),
                    "component": label,
                    "dla": float(dla[i, j]),
                    "logit": float(raw[j]),
                    "gap": float(gap[j]),
                    "candidate_mass": candidate_mass,
                    "top_token": top_token,
                    "collision": collision,
                })

    results = pd.DataFrame(rows)
    out_path = output_path(DATA_DIR, OUT_STEM, ".parquet")
    results.to_parquet(out_path)
    write_run_info(out_path, script=Path(__file__).name, prompts=[PROMPT], model=MODEL, challenge=CHALLENGE)
    print(f"{len(results)} rows -> {out_path}")


if __name__ == "__main__":
    main()
