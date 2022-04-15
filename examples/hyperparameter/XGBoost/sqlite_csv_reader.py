import sqlite3
import pandas as pd
from qlib.log import get_module_logger

logger = get_module_logger("CSV database connection")

db_file = "./db.sqlite3"


logger.info("Connecting to database...")
conn = sqlite3.connect(db_file, isolation_level=None,
                       detect_types=sqlite3.PARSE_COLNAMES)

def viewing_all_tables(db_file, conn):
    c = conn.cursor()
    tables = c.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    for table in tables:
        print("\n------- {} -------".format(table))
        viewing_table(db_file, table[0])
        print("------------------\n")

def viewing_table(db_file, table_name):
    logger.info("Parsing table into dataframe")
    db_df = pd.read_sql_query("SELECT * FROM {}".format(table_name), conn)

    logger.info("Viewing database - {}".format(table_name))
    print(db_df)
    return db_df

viewing_all_tables(db_file, conn)