from pathlib import Path

import pandas as pd

from importlib.resources import files

from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer

from mortgageapplication.probe_locations import create_probes


DEVICE = "cpu"

load_dotenv()

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.10.hook_resid_pre"
)

model = HookedSAETransformer.from_pretrained_no_processing(
    "gpt2",
    device=DEVICE,
    **sae.cfg.metadata.model_from_pretrained_kwargs
)


def load_prompt_template(name: str) -> str:
    return files("mortgageapplication").joinpath("prompts", name).read_text()


frame = pd.read_csv(
    Path(__file__).parent.parent
    / "data"
    / "challenge_tables_2026-10-20T21:37:50.187269"
    / "0529bcc8-5561-4ba7-b91b-34ad59ddac48.csv"
)

# Insert the "<:" and ":>"
for col in ["first_name", "last_name", "city"]:
    frame[col] = "<:" + frame[col] + ":>"

prompt = load_prompt_template("loan_prioritization.txt").format(
    table=frame.rename_axis("id").to_csv(),
    n=len(frame),
    example="4,2",
)

probes = create_probes(prompt, model)

for probe in probes:
    print(probe.string)

# What do we do next?
