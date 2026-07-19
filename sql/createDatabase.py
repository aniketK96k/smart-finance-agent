"""
One-time (safe to re-run) schema setup for the trading/ledger tables,
including pending_orders for tracking unfilled orders placed while
markets were closed.

Run this once before wiring buy_or_sell_stock into your agent:
    python setup_trading_tables.py
"""

import sqlite3
import os

DB_PATH = "F:/GenAi/ProjectFInancial/data/finance.db"

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ---------------------------------------------------------
# 1. Holdings — current stock positions
# ---------------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS holdings(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    quantity REAL NOT NULL DEFAULT 0,
    avg_price REAL NOT NULL DEFAULT 0,
    total_invested REAL NOT NULL DEFAULT 0
)
""")

# ---------------------------------------------------------
# 2. Ledger — append-only audit trail of every money movement
# ---------------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS ledger(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL DEFAULT (datetime('now')),
    type TEXT NOT NULL CHECK(type IN ('credit', 'debit')),
    category TEXT NOT NULL,          -- 'stock_buy', 'stock_sell', 'deposit', 'withdrawal', 'other'
    amount REAL NOT NULL,
    description TEXT,
    balance_after REAL NOT NULL
)
""")

# ---------------------------------------------------------
# 3. Wallet — single-row running cash balance
# ---------------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS wallet(
    id INTEGER PRIMARY KEY CHECK (id = 1),
    balance REAL NOT NULL DEFAULT 0
)
""")
cursor.execute("INSERT OR IGNORE INTO wallet (id, balance) VALUES (1, 0)")

# ---------------------------------------------------------
# 4. Pending orders — trades placed but not yet filled
#    (e.g. market closed). Row is deleted once reconciled.
# ---------------------------------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS pending_orders(
    order_id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    quantity REAL NOT NULL,
    placed_at TEXT NOT NULL DEFAULT (datetime('now'))
)
""")

conn.commit()

# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables in finance.db:", [r[0] for r in cursor.fetchall()])

cursor.execute("SELECT balance FROM wallet WHERE id = 1")
print("Wallet balance:", cursor.fetchone()[0])

conn.close()
print("Schema ready.")