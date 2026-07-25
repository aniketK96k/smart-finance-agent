# sql/risk.py
import sqlite3
from pydantic import BaseModel
from langchain_core.tools import tool

DB_PATH = "F:/GenAi/ProjectFInancial/data/finance.db"


def check_position_risk(symbol: str, dollar_amount: float, sector: str | None = None,
                         max_position_pct: float = 0.20, max_sector_pct: float = 0.40) -> dict:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT balance FROM wallet WHERE id = 1")
    cash = cursor.fetchone()[0]

    cursor.execute("SELECT symbol, quantity, avg_price, sector FROM holdings")
    holdings = cursor.fetchall()
    conn.close()

    holdings_value = sum(qty * price for _, qty, price, _ in holdings)
    portfolio_value = cash + holdings_value
    if portfolio_value == 0:
        return {"approved": True, "flags": []}

    existing = sum(qty * price for sym, qty, price, _ in holdings if sym == symbol)
    position_pct = (existing + dollar_amount) / portfolio_value

    sector_value = sum(qty * price for _, qty, price, sec in holdings if sec == sector) if sector else 0
    sector_pct = (sector_value + dollar_amount) / portfolio_value if sector else 0

    flags = []
    if position_pct > max_position_pct:
        flags.append(f"Position would be {position_pct:.1%} of portfolio (limit {max_position_pct:.0%})")
    if sector and sector_pct > max_sector_pct:
        flags.append(f"Sector exposure would be {sector_pct:.1%} (limit {max_sector_pct:.0%})")
    if dollar_amount > cash:
        flags.append(f"Insufficient balance: have {cash}, need {dollar_amount}")

    return {"approved": len(flags) == 0, "flags": flags}


class DiversificationInput(BaseModel):
    pass

@tool(args_schema=DiversificationInput)
async def get_portfolio_diversification() -> dict:
    """Returns sector breakdown and concentration risk for the current
    portfolio. Use when the user asks about diversification, sector
    exposure, or how balanced their holdings are."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, quantity, avg_price, sector FROM holdings")
    rows = cursor.fetchall()
    conn.close()

    total = sum(qty * price for _, qty, price, _ in rows)
    by_sector = {}
    for _, qty, price, sector in rows:
        key = sector or "Unclassified"
        by_sector[key] = by_sector.get(key, 0) + qty * price

    breakdown = {s: round(v / total, 4) for s, v in by_sector.items()} if total else {}
    most_concentrated = max(breakdown.items(), key=lambda x: x[1]) if breakdown else None

    return {
        "sector_breakdown": breakdown,
        "most_concentrated": most_concentrated,
        "total_portfolio_value": round(total, 2),
    }