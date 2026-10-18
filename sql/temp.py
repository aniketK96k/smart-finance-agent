# run_once_add_sector_column.py
import sqlite3

conn = sqlite3.connect("F:/GenAi/ProjectFInancial/data/finance.db")
conn.execute("ALTER TABLE holdings ADD COLUMN sector TEXT")
conn.commit()
conn.close()
print("done")