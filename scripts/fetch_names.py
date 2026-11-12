import pandas as pd
from sqlalchemy import text, create_engine


def main():
    db = create_engine("postgresql+psycopg://michael@localhost:5432/michael")
    
    q = """
    SELECT name
    FROM hmda.rosenman_last_name_race;
    """

    frame = pd.read_sql(q, db)
    frame.to_parquet("data/last_names.parquet")

    q = """
    SELECT name
    FROM hmda.rosenman_first_name_race;
    """

    frame = pd.read_sql(q, db)
    frame.to_parquet("data/first_names.parquet")


if __name__ == "__main__":
    main()
