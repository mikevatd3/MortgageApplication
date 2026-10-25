from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import re
import tqdm

from mortgageapplication import load_prompt_template

CHALLENGE = "challenge_tables_2026-10-23_full"
DATA_DIR = Path(__file__).parent.parent / "data" / CHALLENGE
MODEL = "google/gemma-2-9b"
OUT_PATH = DATA_DIR / "rankings__completion.csv"

INT_LIST = re.compile(r"\d+(,\d+)*")

CONST_COLS = ["credit_score", "property_value", "loan_amount", "income", "debt_to_income"]
CITY_ONLY_COLS = ["city", "state"] + CONST_COLS
NAME_ONLY_COLS = ["first_name", "last_name"] + CONST_COLS


