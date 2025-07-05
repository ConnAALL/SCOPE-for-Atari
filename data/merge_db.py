"""
Script for merging two databases into a new one

Usage:
    python merge_db.py primary.db secondary.db output.db
"""

import sqlite3
import shutil
import os
import argparse


def get_non_pk_columns(cur: sqlite3.Cursor,
                       table_name: str) -> list[str]:
    """Returns a list of column names in `table_name` where the primary key is not set"""
    cur.execute(f"PRAGMA table_info('{table_name}')")
    cols = cur.fetchall()  # each row: (cid, name, type, notnull, dflt_value, pk)
    return [row[1] for row in cols if row[5] == 0]


def merge_databases(primary_path: str,
                    secondary_path: str,
                    output_path: str):
    """Merge two databases into a new one"""
    # Copy the primary database to the output path
    shutil.copyfile(primary_path, output_path)

    # Open the output database
    conn = sqlite3.connect(output_path)
    cur = conn.cursor()

    # Attach the secondary database as 'to_merge'
    cur.execute("ATTACH DATABASE ? AS to_merge", (secondary_path,))

    # List all user tables
    cur.execute("""
        SELECT name
          FROM sqlite_master
         WHERE type='table'
           AND name NOT LIKE 'sqlite_%'
    """)
    tables = [row[0] for row in cur.fetchall()]

    # For each table, merge the data from the secondary database into the primary one
    for tbl in tables:
        print(f"[merge_db] Merging table `{tbl}`...")

        # Find columns to copy (exclude primary‐key columns)
        cols = get_non_pk_columns(cur, tbl)
        if not cols:
            print(f"[merge_db] No non-PK columns found in `{tbl}`, skipping.")
            continue

        col_list = ', '.join(f'"{c}"' for c in cols)
        select_list = ', '.join(f'"{c}"' for c in cols)

        # Insert the data from the secondary database into the primary one
        sql = f'''
            INSERT INTO "{tbl}" ({col_list})
            SELECT {select_list}
              FROM to_merge."{tbl}"
        '''
        cur.execute(sql)

    # Commit and detach the secondary database
    conn.commit()
    cur.execute("DETACH DATABASE to_merge")
    conn.close()
    print(f"[merge_db] Merged data written to `{output_path}`")


def main():
    # Getting the command line arguments
    parser = argparse.ArgumentParser(
        description="Merge two SQLite databases into a new one",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s primary.db secondary.db output.db
    %(prog)s runs_0_100.db runs_5000.db combined_runs.db
        """
    )
    
    parser.add_argument(
        "primary_db",
        help="Path to the primary database that will serve as the base"
    )
    parser.add_argument(
        "secondary_db",
        help="Path to the secondary database to merge into the primary one"
    )
    parser.add_argument(
        "output_db",
        help="Path where the merged database will be created"
    )

    args = parser.parse_args()

    # Validate input files exist
    for path in (args.primary_db, args.secondary_db):
        if not os.path.isfile(path):
            parser.error(f"[merge_db] File does not exist: {path}")

    # Validate output is different from inputs
    if os.path.abspath(args.output_db) in map(os.path.abspath, (args.primary_db, args.secondary_db)):
        parser.error("[merge_db] Output database must be different from the input databases")

    merge_databases(args.primary_db, args.secondary_db, args.output_db)


if __name__ == "__main__":
    main()
