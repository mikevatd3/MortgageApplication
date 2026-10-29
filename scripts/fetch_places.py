"""Export places with their racial composition from the queueprompt database.

Run with queueprompt's environment (it owns the DB driver and credentials):

    uv run --project ../queueprompt python scripts/fetch_places.py
"""
from pathlib import Path
import pandas as pd
from sqlalchemy import text

from queueprompt.cities import tidy_places
from queueprompt.config import load_db

MIN_POP = 150_000  # same pool the challenge tables draw cities from
OUT_PATH = Path(__file__).parent.parent / "data" / "places_2026-10-29.csv"


def main():
    q = text("""
    SELECT place_geoid AS geoid,
           TRIM(SPLIT_PART(place_name, ',', 1)) AS city,
           TRIM(SPLIT_PART(place_name, ',', 2)) AS state,
           total_pop,
           black::NUMERIC    / total_pop AS bla,
           white::NUMERIC    / total_pop AS whi,
           asian::NUMERIC    / total_pop AS asi,
           hispanic::NUMERIC / total_pop AS his,
           (total_pop - (black + white + asian + hispanic))::NUMERIC / total_pop AS oth
    FROM hmda.hmda_place_race
    WHERE total_pop >= :min_pop
    AND place_name NOT LIKE '%, Puerto Rico'
    ORDER BY place_geoid
    """)

    frame = pd.read_sql(q, load_db(), params={"min_pop": MIN_POP})
    frame = tidy_places(frame)
    frame.to_csv(OUT_PATH, index=False)
    print(f"{len(frame)} places -> {OUT_PATH}")


if __name__ == "__main__":
    main()
