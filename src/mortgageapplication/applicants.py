"""Shaping an applicant table before it goes into a prompt."""
import pandas as pd

NAME_COLS = ["last_name", "first_name"]


def last_name_first(frame: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns so last_name comes before first_name, wherever the name columns sit."""
    cols = list(frame.columns)
    at = cols.index("first_name")
    cols = [c for c in cols if c not in NAME_COLS]
    cols[at:at] = NAME_COLS
    return frame[cols]


def title_case_names(frame: pd.DataFrame) -> pd.DataFrame:
    """Title-case the name columns ("MCNIER" -> "Mcnier"), the way names appear in the other prompts."""
    frame = frame.copy()
    for col in NAME_COLS:
        frame[col] = frame[col].str.title()
    return frame
