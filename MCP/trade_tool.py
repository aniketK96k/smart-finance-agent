# trade_tools.py
import sqlite3
import asyncio
import copy
from langchain_core.tools import tool

from sql.risk import check_position_risk

DB_PATH = "F:/GenAi/ProjectFInancial/data/finance.db"


def _get_alpaca_tool(tools, name: str):
    return next(t for t in tools if t.name == name)


def _record_trade(symbol: str, side: str, quantity: float, price: float, order_id: str = ""):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT balance FROM wallet WHERE id = 1")
        balance = cursor.fetchone()[0]
        total = quantity * price
        tag = f" [order:{order_id}]" if order_id else ""

        if side == "buy":
            if balance < total:
                raise ValueError(f"Insufficient balance: have {balance}, need {total}")
            new_balance = balance - total
            cursor.execute("SELECT quantity, total_invested FROM holdings WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if row:
                new_qty = row[0] + quantity
                new_invested = row[1] + total
                cursor.execute(
                    "UPDATE holdings SET quantity=?, avg_price=?, total_invested=? WHERE symbol=?",
                    (new_qty, new_invested / new_qty, new_invested, symbol),
                )
            else:
                cursor.execute(
                    "INSERT INTO holdings (symbol, quantity, avg_price, total_invested) VALUES (?,?,?,?)",
                    (symbol, quantity, price, total),
                )
            cursor.execute(
                "INSERT INTO ledger (type, category, amount, description, balance_after) VALUES (?,?,?,?,?)",
                ("debit", "stock_buy", total, f"Bought {quantity} {symbol} @ {price}{tag}", new_balance),
            )
        else:  # sell
            cursor.execute("SELECT quantity, avg_price, total_invested FROM holdings WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            if not row or row[0] < quantity:
                raise ValueError(f"Not enough {symbol} to sell")
            new_qty = row[0] - quantity
            cost_sold = row[1] * quantity
            if new_qty == 0:
                cursor.execute("DELETE FROM holdings WHERE symbol = ?", (symbol,))
            else:
                cursor.execute(
                    "UPDATE holdings SET quantity=?, total_invested=? WHERE symbol=?",
                    (new_qty, row[2] - cost_sold, symbol),
                )
            new_balance = balance + total
            cursor.execute(
                "INSERT INTO ledger (type, category, amount, description, balance_after) VALUES (?,?,?,?,?)",
                ("credit", "stock_sell", total, f"Sold {quantity} {symbol} @ {price}{tag}", new_balance),
            )

        cursor.execute("UPDATE wallet SET balance = ? WHERE id = 1", (new_balance,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _order_already_recorded(order_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM ledger WHERE description LIKE ?", (f"%[order:{order_id}]%",))
    row = cursor.fetchone()
    conn.close()
    return row is not None


def _add_pending_order(order_id: str, symbol: str, side: str, quantity: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO pending_orders (order_id, symbol, side, quantity) VALUES (?,?,?,?)",
        (order_id, symbol, side, quantity),
    )
    conn.commit()
    conn.close()


def _remove_pending_order(order_id: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM pending_orders WHERE order_id = ?", (order_id,))
    conn.commit()
    conn.close()


def _get_all_pending_order_ids():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT order_id FROM pending_orders")
    ids = [row[0] for row in cursor.fetchall()]
    conn.close()
    return ids


def _extract_field(result, *keys):
    """
    Alpaca MCP tool results can come back as a dict, a pydantic-like object,
    or a list of content blocks (depending on the MCP adapter version).
    This normalizes access across those shapes.
    """
    if result is None:
        return None

    if isinstance(result, list):
        for block in result:
            val = _extract_field(block, *keys)
            if val is not None:
                return val
        return None

    if isinstance(result, dict):
        for key in keys:
            if key in result and result[key] is not None:
                return result[key]
        return None

    for key in keys:
        if hasattr(result, key):
            val = getattr(result, key)
            if val is not None:
                return val
    return None


def build_trade_tools(alpaca_tools):
    """
    IMPORTANT: pass the RAW (unsanitized) Alpaca tools here — these tools call
    .ainvoke() directly with hand-built dicts, bypassing Gemini's function-calling
    schema entirely, so sanitization is irrelevant for them.

    We deep-copy the incoming list so sanitizing a filtered copy of alpaca_tools
    elsewhere (for the LLM-facing read-only tools) can't mutate the objects
    this function already grabbed references to.
    """
    alpaca_tools = copy.deepcopy(alpaca_tools)

    place_order = _get_alpaca_tool(alpaca_tools, "place_stock_order")
    get_order = _get_alpaca_tool(alpaca_tools, "get_order_by_id")

    @tool
    async def buy_or_sell_stock(symbol: str, side: str, quantity: float,
                                 sector: str | None = None, force: bool = False) -> dict:
        """
        Place a market buy or sell order for a stock via Alpaca and record it
        in the local wallet/holdings/ledger tables if it fills immediately.
        side must be 'buy' or 'sell'. Always use this tool for trading —
        never call place_stock_order directly.

        Before a BUY, if the position or sector risk limits would be
        breached, this returns status "needs_confirmation" instead of
        placing the order. Only set force=True after the user has
        explicitly confirmed proceeding — phrases like "yes", "proceed
        anyway", "confirm", "do it" following a needs_confirmation
        response all count as confirmation.

        If the order does not fill right away (e.g. market closed), it is
        queued for automatic reconciliation — the local wallet is NOT touched
        until sync_pending_orders (or check_and_record_pending_order) confirms
        the fill.
        """
        side = side.lower().strip()
        if side not in ("buy", "sell"):
            return {"status": "error", "message": "side must be 'buy' or 'sell'"}

        symbol = symbol.upper().strip()

        if side == "buy" and not force:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT avg_price FROM holdings WHERE symbol = ?", (symbol,))
            row = cursor.fetchone()
            conn.close()
            est_price = row[0] if row else None

            if est_price:
                risk_check = check_position_risk(symbol, quantity * est_price, sector)
                if not risk_check["approved"]:
                    return {
                        "status": "needs_confirmation",
                        "flags": risk_check["flags"],
                        "message": "This trade breaches your risk limits. Confirm to proceed anyway.",
                    }

        order = await place_order.ainvoke({
            "symbol": symbol,
            "side": side,
            "qty": quantity,
            "type": "market",
            "time_in_force": "day",
        })
        # ... rest unchanged

        fill_price = _extract_field(order, "filled_avg_price")
        order_id = _extract_field(order, "id", "order_id")
        status = _extract_field(order, "status")

        if status in ("rejected", "canceled", "expired"):
            return {"status": "order_failed", "order_id": order_id, "order_status": status, "raw": order}

        # Market orders can take a moment to fill — poll briefly if not filled yet
        for _ in range(5):
            if fill_price:
                break
            await asyncio.sleep(1)
            check = await get_order.ainvoke({"order_id": order_id})
            fill_price = _extract_field(check, "filled_avg_price")
            status = _extract_field(check, "status") or status

        if not fill_price:
            _add_pending_order(order_id, symbol, side, quantity)
            return {
                "status": "pending_not_recorded",
                "order_id": order_id,
                "order_status": status,
                "note": (
                    "Order queued for automatic reconciliation. "
                    "Call sync_pending_orders anytime to check all pending trades — "
                    "no need to track this order_id yourself."
                ),
            }

        _record_trade(symbol, side, quantity, float(fill_price), order_id=order_id)
        return {
            "status": "success",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "fill_price": fill_price,
            "order_id": order_id,
        }

    @tool
    async def check_and_record_pending_order(order_id: str) -> dict:
        """
        Check a previously placed order that was pending/unfilled, and if it has
        since filled, record it in the local wallet/holdings/ledger tables and
        remove it from the pending queue. Safe to call multiple times — does
        nothing if already recorded or still unfilled. Rejected/canceled/expired
        orders are also removed from the pending queue.
        """
        order = await get_order.ainvoke({"order_id": order_id})

        status = _extract_field(order, "status")
        fill_price = _extract_field(order, "filled_avg_price")
        symbol = _extract_field(order, "symbol")
        side = _extract_field(order, "side")
        filled_qty = _extract_field(order, "filled_qty", "qty")

        if status in ("rejected", "canceled", "expired"):
            _remove_pending_order(order_id)  # nothing more to wait for
            return {"status": "order_failed", "order_id": order_id, "order_status": status}

        if not fill_price:
            return {"status": "still_pending", "order_id": order_id, "order_status": status}

        if _order_already_recorded(order_id):
            _remove_pending_order(order_id)  # stale row cleanup, just in case
            return {"status": "already_recorded", "order_id": order_id}

        _record_trade(
            symbol.upper().strip(),
            side.lower().strip(),
            float(filled_qty),
            float(fill_price),
            order_id=order_id,
        )
        _remove_pending_order(order_id)
        return {
            "status": "success",
            "symbol": symbol,
            "side": side,
            "quantity": filled_qty,
            "fill_price": fill_price,
            "order_id": order_id,
        }

    @tool
    async def sync_pending_orders() -> dict:
        """
        Check every order still marked pending and record any that have since
        filled (removing failed/canceled ones from the queue too). Use this
        whenever the user asks something like 'has my order gone through',
        'update my portfolio', or 'sync my trades' without naming a specific
        order_id.
        """
        ids = _get_all_pending_order_ids()
        if not ids:
            return {"status": "no_pending_orders"}

        results = []
        for oid in ids:
            result = await check_and_record_pending_order.ainvoke({"order_id": oid})
            results.append(result)
        return {"status": "synced", "checked": len(ids), "results": results}

    return [buy_or_sell_stock, check_and_record_pending_order, sync_pending_orders]