import sqlite3

conn = sqlite3.connect("data/finance.db")

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS sales(
    id INTEGER PRIMARY KEY,
    quarter TEXT,
    revenue REAL,
    expenses REAL,
    profit REAL
)
""")

cursor.executemany(
    """
    INSERT INTO sales
    (quarter,revenue,expenses,profit)
    VALUES(?,?,?,?)
    """,
    [
        ("Q1",100000,70000,30000),
        ("Q2",90000,75000,15000),
        ("Q3",120000,80000,40000),
        ("Q4",150000,90000,60000),
    ]
)

conn.commit()
conn.close()

print("Database Created")