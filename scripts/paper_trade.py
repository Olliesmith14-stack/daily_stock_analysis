#!/usr/bin/env python3
"""
Paper-trading ledger: follow the daily analysis calls with demo money.

Runs after `python main.py` in the daily workflow. It keeps three portfolios
in GBP, all starting from the same cash:

  followed : starts in cash, trades only on the analyser's calls
  held     : your stock list bought on day one and never traded
  bench    : everything in the benchmark (default VUAG.L)

Comparing them answers two questions: did your picks beat the index
(held vs bench), and did following the calls help (followed vs held)?

Realism (defaults model a UK investor on Trading 212, in a stocks & shares ISA):
  - Each order fills at its market's first close after the call was made
    (London 16:30 UK time, New York 16:00 ET), so a price the analysis had
    already seen is never used. Orders wait until that close is available.
  - 0.15% FX fee on every buy and sell of a non-GBP stock.
  - 0.5% stamp duty on buying UK shares (not ETFs).
  - Fractional shares (set PAPER_UK_FRACTIONAL=false for whole UK shares).
  - Interest on uninvested cash, accrued daily (PAPER_CASH_INTEREST).
  - Dividends paid in cash on the ex-date, US ones net of 15% withholding
    tax (W-8BEN rate). No UK tax (ISA).

Rules for `followed` (a "slot" = equity x weight / total weight):
  buy    -> top the holding up to one slot, limited by cash
  hold   -> nothing (Watch / Hold)
  reduce -> sell half
  sell   -> sell all

The ledger is a JSON file (PAPER_LEDGER_PATH). The workflow keeps it
encrypted on a separate branch so it is not public; see
scripts/paper_trade_run.sh and docs/paper-trading.md.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("paper_trade")

# Public Actions logs: never log holdings, values or trades, only counts.


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class Config:
    start_cash: float = 10_000.0
    platform: str = "Trading 212"
    fx_fee: float = 0.0015
    cash_interest: float = 0.0355
    uk_fractional: bool = True
    stamp_duty: float = 0.005
    us_withholding: float = 0.15
    benchmark: str = "VUAG.L"
    weights: Dict[str, float] = None
    universe: List[str] = None
    ledger_path: Path = ROOT / "data" / "paper_ledger.json"
    lookback_hours: int = 12

    @classmethod
    def from_env(cls) -> "Config":
        def f(name, default):
            v = os.getenv(name, "").strip()
            return float(v) if v else default

        weights = {}
        for part in os.getenv("PAPER_WEIGHTS", "").split(","):
            if ":" in part:
                sym, w = part.split(":", 1)
                weights[sym.strip().upper()] = float(w)
        universe = [s for s in (norm_symbol(x) for x in os.getenv("STOCK_LIST", "").split(",")) if s]
        return cls(
            start_cash=f("PAPER_START_CASH", 10_000.0),
            platform=os.getenv("PAPER_PLATFORM", "").strip() or "Trading 212",
            fx_fee=f("PAPER_FX_FEE", 0.0015),
            cash_interest=f("PAPER_CASH_INTEREST", 0.0355),
            uk_fractional=os.getenv("PAPER_UK_FRACTIONAL", "true").strip().lower() not in ("false", "0", "no"),
            stamp_duty=f("PAPER_STAMP_DUTY", 0.005),
            us_withholding=f("PAPER_US_WITHHOLDING", 0.15),
            benchmark=(os.getenv("PAPER_BENCHMARK", "").strip() or "VUAG.L").upper(),
            weights=weights,
            universe=universe,
            ledger_path=Path(os.getenv("PAPER_LEDGER_PATH", "").strip() or ROOT / "data" / "paper_ledger.json"),
            lookback_hours=int(f("PAPER_LOOKBACK_HOURS", 12)),
        )

    def weight(self, sym: str) -> float:
        return self.weights.get(sym, 1.0) if self.weights else 1.0


SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9\-]*(\.[A-Z]{1,3})?$")


def norm_symbol(code: Optional[str]) -> Optional[str]:
    """Yahoo-style ticker or None (A-shares/HK codes etc. are skipped)."""
    sym = (code or "").strip().upper()
    return sym if SYMBOL_RE.match(sym) else None


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

@dataclass
class Signal:
    symbol: str
    score: Optional[int]
    advice: str
    action: str  # buy | hold | reduce | sell


def to_action(advice: Optional[str], score: Optional[int]) -> str:
    """operation_advice first (it includes the analyser's guardrails), score second."""
    text = (advice or "").strip().lower()
    if text:
        if any(k in text for k in ("sell", "exit", "卖出", "清仓")):
            return "sell"
        if any(k in text for k in ("reduce", "trim", "减")):
            return "reduce"
        if any(k in text for k in ("buy", "add", "买", "加仓")):
            return "buy"
        if any(k in text for k in ("hold", "watch", "wait", "持有", "观望", "观察")):
            return "hold"
    if score is None:
        return "hold"
    if score >= 60:
        return "buy"
    if score >= 40:
        return "hold"
    if score >= 20:
        return "reduce"
    return "sell"


def load_signals(lookback_hours: int, allowed: Optional[set] = None) -> List[Signal]:
    """Latest stock analysis per code saved in the last `lookback_hours`.

    Only codes in `allowed` (the STOCK_LIST) are used when it is given, so the
    market-review entry and anything else that is not one of your stocks is ignored.
    """
    from src.storage import AnalysisHistory, get_db

    since = datetime.now() - timedelta(hours=lookback_hours)
    latest: Dict[str, Signal] = {}
    with get_db().session_scope() as session:
        rows = (
            session.query(AnalysisHistory)
            .filter(AnalysisHistory.created_at >= since)
            .order_by(AnalysisHistory.created_at.asc())
            .all()
        )
        for row in rows:
            if row.operation_advice is None and row.sentiment_score is None:
                continue
            sym = norm_symbol(row.code)
            if sym is None or (allowed and sym not in allowed):
                continue
            latest[sym] = Signal(sym, row.sentiment_score, row.operation_advice or "",
                                 to_action(row.operation_advice, row.sentiment_score))
    return list(latest.values())


# --------------------------------------------------------------------------
# Market data (Yahoo Finance via yfinance; replaceable in tests)
# --------------------------------------------------------------------------

class MarketData:
    """closes/dividends keyed by ISO date string, in the listing currency."""

    def __init__(self):
        self._cache = {}

    def _history(self, sym: str, start: str):
        key = (sym, start)
        if key not in self._cache:
            import yfinance as yf

            df = yf.Ticker(sym).history(start=start, auto_adjust=False, actions=True)
            closes, divs = {}, {}
            for ts, row in df.iterrows():
                d = ts.date().isoformat()
                if row.get("Close") == row.get("Close"):  # not NaN
                    closes[d] = float(row["Close"])
                if float(row.get("Dividends", 0) or 0) > 0:
                    divs[d] = float(row["Dividends"])
            self._cache[key] = (closes, divs)
        return self._cache[key]

    def closes(self, sym: str, start: str) -> Dict[str, float]:
        return self._history(sym, start)[0]

    def dividends(self, sym: str, start: str) -> Dict[str, float]:
        return self._history(sym, start)[1]

    def meta(self, sym: str) -> dict:
        import yfinance as yf

        t = yf.Ticker(sym)
        try:
            currency = t.fast_info.get("currency")
        except Exception:
            currency = None
        currency = currency or ("GBp" if sym.endswith(".L") else "USD")
        etf, name = False, None
        try:
            info = t.info or {}
            etf = info.get("quoteType") == "ETF"
            name = info.get("shortName") or info.get("longName")
        except Exception:
            pass
        return {"currency": currency, "etf": etf, "name": name}


def on_or_before(series: Dict[str, float], d: str) -> Optional[Tuple[str, float]]:
    keys = [k for k in series if k <= d]
    if not keys:
        return None
    k = max(keys)
    return k, series[k]


def first_on_or_after(series: Dict[str, float], d: str, before: str) -> Optional[Tuple[str, float]]:
    keys = [k for k in series if d <= k < before]
    if not keys:
        return None
    k = min(keys)
    return k, series[k]


class Pricer:
    """Converts listing-currency prices to GBP."""

    def __init__(self, md: MarketData, meta: Dict[str, dict], start: str):
        self.md, self.meta, self.start = md, meta, start
        self._fx: Dict[str, Dict[str, float]] = {}

    def ccy(self, sym: str) -> str:
        c = self.meta[sym]["currency"]
        return "GBP" if c in ("GBp", "GBX", "GBP") else c.upper()

    def unit(self, sym: str) -> float:
        return 0.01 if self.meta[sym]["currency"] in ("GBp", "GBX") else 1.0

    def fx(self, ccy: str, d: str) -> float:
        """Units of `ccy` per 1 GBP on/before date d."""
        if ccy == "GBP":
            return 1.0
        if ccy not in self._fx:
            self._fx[ccy] = self.md.closes(f"GBP{ccy}=X", self.start)
        hit = on_or_before(self._fx[ccy], d) or first_on_or_after(self._fx[ccy], d, "9999")
        if not hit:
            raise RuntimeError(f"No GBP{ccy} rate for {d}")
        return hit[1]

    def to_gbp(self, sym: str, local_price: float, d: str) -> float:
        return local_price * self.unit(sym) / self.fx(self.ccy(sym), d)

    def gbp_on_or_before(self, sym: str, d: str) -> Optional[float]:
        hit = on_or_before(self.md.closes(sym, self.start), d)
        return self.to_gbp(sym, hit[1], hit[0]) if hit else None


# --------------------------------------------------------------------------
# Portfolio mechanics
# --------------------------------------------------------------------------

def new_port(cash: float) -> dict:
    return {"cash": cash, "holdings": {}, "cost": {}, "fees": 0.0, "dividends": 0.0, "interest": 0.0}


class Costs:
    def __init__(self, cfg: Config, pricer: Pricer):
        self.cfg, self.p = cfg, pricer

    def is_uk_share(self, sym: str) -> bool:
        return sym.endswith(".L") and not self.p.meta[sym].get("etf")

    def whole_shares(self, sym: str) -> bool:
        return self.p.ccy(sym) != "USD" and not self.cfg.uk_fractional

    def buy_rate(self, sym: str) -> float:
        r = self.cfg.fx_fee if self.p.ccy(sym) != "GBP" else 0.0
        return r + (self.cfg.stamp_duty if self.is_uk_share(sym) else 0.0)

    def sell_rate(self, sym: str) -> float:
        return self.cfg.fx_fee if self.p.ccy(sym) != "GBP" else 0.0


def port_value(port: dict, prices: Dict[str, float]) -> float:
    return port["cash"] + sum(q * prices.get(s, 0.0) for s, q in port["holdings"].items())


def do_buy(port, sym, budget, price, costs: Costs) -> Optional[dict]:
    rate = costs.buy_rate(sym)
    budget = min(budget, port["cash"])
    qty = budget / (price * (1 + rate))
    qty = math.floor(qty) if costs.whole_shares(sym) else math.floor(qty * 1e4) / 1e4
    if qty <= 0:
        return None
    gross = qty * price
    fee = gross * rate
    port["cash"] -= gross + fee
    port["holdings"][sym] = port["holdings"].get(sym, 0.0) + qty
    port["cost"][sym] = port["cost"].get(sym, 0.0) + gross + fee
    port["fees"] += fee
    return {"side": "BUY", "symbol": sym, "qty": qty, "price": price, "value": gross, "fee": fee}


def do_sell(port, sym, qty, price, costs: Costs) -> Optional[dict]:
    held = port["holdings"].get(sym, 0.0)
    qty = min(qty, held)
    if qty <= 0:
        return None
    gross = qty * price
    fee = gross * costs.sell_rate(sym)
    port["cash"] += gross - fee
    port["cost"][sym] = port["cost"].get(sym, 0.0) * (1 - qty / held)
    port["holdings"][sym] = held - qty
    if port["holdings"][sym] <= 1e-9:
        port["holdings"].pop(sym)
        port["cost"].pop(sym, None)
    port["fees"] += fee
    return {"side": "SELL", "symbol": sym, "qty": qty, "price": price, "value": gross, "fee": fee}


def credit_dividends(state, md: MarketData, pricer: Pricer, cfg: Config, after: str, through: str) -> int:
    """Pay dividends with ex-date in (after, through] on holdings as they stand."""
    paid = 0
    syms = set()
    for name in ("followed", "held", "bench"):
        syms |= set(state[name]["holdings"])
    for sym in sorted(syms):
        divs = md.dividends(sym, state["data_start"])
        closes = md.closes(sym, state["data_start"])
        for d, amt in sorted(divs.items()):
            if not (after < d <= through):
                continue
            # Guard against Yahoo pence/pound mix-ups on UK dividends.
            px = on_or_before(closes, d)
            if px and amt > 0.25 * px[1]:
                logger.warning("Skipping implausible dividend for one holding")
                continue
            if px and pricer.unit(sym) == 0.01 and amt < 0.0005 * px[1]:
                amt *= 100
            per_share = pricer.to_gbp(sym, amt, d)
            if pricer.ccy(sym) == "USD":
                per_share *= 1 - cfg.us_withholding
            for name in ("followed", "held", "bench"):
                port = state[name]
                q = port["holdings"].get(sym, 0.0)
                if q > 0:
                    cash = q * per_share
                    port["cash"] += cash
                    port["dividends"] += cash
                    if name == "followed":
                        state["dividend_log"].append({"date": d, "symbol": sym, "amount": round(cash, 2)})
                    paid += 1
    return paid


def accrue_interest(state, cfg: Config, through: str) -> None:
    """Daily interest on uninvested cash from interest_through to `through`."""
    since = state.get("interest_through")
    if not since or through <= since or cfg.cash_interest <= 0:
        state["interest_through"] = max(filter(None, [since, through]))
        return
    days = (date.fromisoformat(through) - date.fromisoformat(since)).days
    factor = (1 + cfg.cash_interest) ** (days / 365) - 1
    for name in ("followed", "held", "bench"):
        port = state[name]
        earned = max(port["cash"], 0.0) * factor
        port["cash"] += earned
        port["interest"] = port.get("interest", 0.0) + earned
    state["interest_through"] = through


# --------------------------------------------------------------------------
# Daily run
# --------------------------------------------------------------------------

def init_state(cfg: Config, today: str) -> dict:
    return {
        "version": 1,
        "created": today,
        "start_date": None,
        "data_start": (date.fromisoformat(today) - timedelta(days=10)).isoformat(),
        "config": {"start_cash": cfg.start_cash, "platform": cfg.platform, "fx_fee": cfg.fx_fee,
                   "cash_interest": cfg.cash_interest, "uk_fractional": cfg.uk_fractional,
                   "stamp_duty": cfg.stamp_duty, "us_withholding": cfg.us_withholding,
                   "benchmark": cfg.benchmark},
        "followed": new_port(cfg.start_cash),
        "held": new_port(cfg.start_cash),
        "bench": new_port(cfg.start_cash),
        "meta": {},
        "queue": [],
        "div_through": None,
        "interest_through": None,
        "closes": {},
        "calls": [],
        "history": {},
        "trades": [],
        "dividend_log": [],
    }


CLOSE_TIMES = {"L": ("Europe/London", 16, 35)}
DEFAULT_CLOSE = ("America/New_York", 16, 5)


def close_instant(sym: str, d: str) -> datetime:
    """When the closing price for `sym` on date `d` becomes final (with a few minutes' margin)."""
    suffix = sym.rsplit(".", 1)[1] if "." in sym else ""
    tz, h, m = CLOSE_TIMES.get(suffix, DEFAULT_CLOSE)
    return datetime.combine(date.fromisoformat(d), time(h, m), ZoneInfo(tz))


def execution_close(closes: Dict[str, float], sym: str, signal_ts: datetime,
                    now: datetime) -> Optional[Tuple[str, float]]:
    """First close after the call; None if it hasn't happened yet."""
    for d in sorted(closes):
        at = close_instant(sym, d)
        if at > signal_ts:
            return (d, closes[d]) if at <= now else None
    return None


def settle(state, cfg: Config, md: MarketData, pend: dict, now: datetime) -> Tuple[List[dict], Optional[str], str]:
    """Fill one queued batch at each stock's first close after the call. Returns (trades, fill_date, status)."""
    sig_date = pend["signal_date"]
    signal_ts = datetime.fromisoformat(pend["signal_ts"])
    stale = now - signal_ts > timedelta(days=7)

    pricer = Pricer(md, state["meta"], state["data_start"])
    costs = Costs(cfg, pricer)
    universe = [s for s in pend["universe"] if s in state["meta"]]
    needed = set(universe) | set(state["followed"]["holdings"]) | {cfg.benchmark}

    fills: Dict[str, Tuple[str, float]] = {}
    for sym in needed:
        hit = execution_close(md.closes(sym, state["data_start"]), sym, signal_ts, now)
        if hit:
            fills[sym] = (hit[0], pricer.to_gbp(sym, hit[1], hit[0]))
        elif not stale:
            return [], None, "waiting"
    if not fills:
        return [], None, "waiting"
    if stale and len(fills) < len(needed):
        logger.warning("%d ticker(s) still had no price after a week and were skipped", len(needed) - len(fills))
        universe = [s for s in universe if s in fills]
    fill_date = max(d for d, _ in fills.values())
    price = {s: p for s, (_, p) in fills.items()}

    # Dividends and interest up to the fill belong to the old holdings and cash.
    if state["div_through"]:
        credit_dividends(state, md, pricer, cfg, state["div_through"], sig_date)
    state["div_through"] = sig_date
    accrue_interest(state, cfg, sig_date)

    for c in pend.get("calls", []):
        if c["symbol"] in fills:
            state["calls"].append({**c, "date": fills[c["symbol"]][0], "price": round(fills[c["symbol"]][1], 6)})

    trades: List[dict] = []
    total_w = sum(cfg.weight(s) for s in universe) or 1.0

    if pend.get("init"):
        state["start_date"] = sig_date
        for sym in universe:
            do_buy(state["held"], sym, cfg.start_cash * cfg.weight(sym) / total_w, price[sym], costs)
        if cfg.benchmark in price:
            do_buy(state["bench"], cfg.benchmark, cfg.start_cash, price[cfg.benchmark], costs)

    port = state["followed"]
    orders = [o for o in pend["orders"] if o["symbol"] in price]
    for o in [o for o in orders if o["action"] in ("sell", "reduce")]:
        held = port["holdings"].get(o["symbol"], 0.0)
        if held <= 0:
            continue
        if o["action"] == "sell":
            qty = held
        elif costs.whole_shares(o["symbol"]):
            qty = max(1, math.floor(held / 2)) if held >= 1 else held
        else:
            qty = held / 2
        t = do_sell(port, o["symbol"], qty, price[o["symbol"]], costs)
        if t:
            trades.append({**t, "reason": f"{o['advice'] or o['action']} ({o['score']})"})

    equity = port_value(port, price)
    buys = sorted([o for o in orders if o["action"] == "buy"], key=lambda o: -(o["score"] or 0))
    for o in buys:
        sym = o["symbol"]
        target = equity * cfg.weight(sym) / total_w
        have = port["holdings"].get(sym, 0.0) * price[sym]
        t = do_buy(port, sym, max(target - have, 0.0), price[sym], costs)
        if t:
            trades.append({**t, "reason": f"{o['advice'] or o['action']} ({o['score']})"})

    for t in trades:
        rounded = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in t.items()}
        state["trades"].append({"date": fills[t["symbol"]][0], **rounded})
    return trades, fill_date, "filled"


def mark_to_market(state, cfg: Config, md: MarketData, today: str) -> Optional[str]:
    pricer = Pricer(md, state["meta"], state["data_start"])
    syms = set(state["meta"])
    last = None
    for sym in syms:
        hit = on_or_before({k: v for k, v in md.closes(sym, state["data_start"]).items() if k < today}, today)
        if hit and (last is None or hit[0] > last):
            last = hit[0]
    if last is None or not state["start_date"]:
        return None
    if state["div_through"] and last > state["div_through"]:
        credit_dividends(state, md, pricer, cfg, state["div_through"], last)
        state["div_through"] = last
    accrue_interest(state, cfg, last)
    closes_log = state.setdefault("closes", {})
    for sym in syms:
        for d, px in md.closes(sym, state["data_start"]).items():
            if state["start_date"] <= d <= last:
                closes_log.setdefault(sym, {})[d] = round(pricer.to_gbp(sym, px, d), 6)
    prices = {s: pricer.gbp_on_or_before(s, last) or 0.0 for s in syms}
    state["last_prices"] = {s: round(p, 6) for s, p in prices.items()}
    state["history"][last] = {n: round(port_value(state[n], prices), 2) for n in ("followed", "held", "bench")}
    return last


def queue_orders(state, cfg: Config, signals: List[Signal], today: str, now: datetime) -> List[Signal]:
    """Queue today's calls. Re-running on the same day replaces that day's batch."""
    universe = list(dict.fromkeys((cfg.universe or []) + [s.symbol for s in signals]
                                  + list(state["followed"]["holdings"])))
    queue = state.setdefault("queue", [])
    if queue and queue[-1]["signal_date"] == today:
        queue.pop()
    init = state["start_date"] is None and not any(b.get("init") for b in queue)
    queue.append({
        "signal_date": today,
        "signal_ts": now.isoformat(),
        "init": init,
        "universe": universe,
        "orders": [{"symbol": s.symbol, "action": s.action, "advice": s.advice, "score": s.score}
                   for s in signals if s.action != "hold"],
        "calls": [{"symbol": s.symbol, "action": s.action, "advice": s.advice, "score": s.score}
                  for s in signals],
    })
    return [s for s in signals if s.action != "hold"]


def ensure_meta(state, md: MarketData, symbols: List[str]) -> List[str]:
    """Load listing details; returns the symbols that have price data (others are skipped)."""
    usable = []
    for sym in symbols:
        try:
            if not md.closes(sym, state["data_start"]):
                raise ValueError("no prices")
            if sym not in state["meta"] or "name" not in state["meta"][sym]:
                state["meta"][sym] = md.meta(sym)
            usable.append(sym)
        except Exception:
            logger.warning("A ticker has no price data on Yahoo Finance and was skipped")
    return usable


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def gbp(x: float) -> str:
    return f"£{x:,.2f}" if abs(x) < 1000 else f"£{x:,.0f}"


def pct(a: float, b: float) -> str:
    return f"{(a / b - 1) * 100:+.1f}%" if b else "n/a"


def summary(state, cfg: Config, trades, fill_date, status, queued: List[Signal], signals: List[Signal]) -> str:
    out = [f"📒 *Paper portfolio* (demo money, {cfg.platform} costs)"]
    if status == "filled":
        if trades:
            out.append(f"\nFilled at the {fill_date} close:")
            for t in trades:
                q = f"{t['qty']:g}"
                out.append(f"• {t['side']} {q} {t['symbol']} @ {gbp(t['price'])} = {gbp(t['value'])}"
                           f" (fees {gbp(t['fee'])}) — {t['reason']}")
        else:
            out.append(f"\nNo trades at the {fill_date} close (no Buy/Reduce/Sell calls to act on).")
    elif status == "waiting":
        out.append("\nEarlier orders are waiting for their market's next close.")

    if signals:
        if queued:
            out.append("\nToday's calls (each fills at its market's next close):")
            for s in queued:
                out.append(f"• {s.symbol}: {s.advice or s.action} ({s.score}) → {s.action.upper()}")
        else:
            out.append("\nToday's calls: all Watch/Hold, no orders.")

    hist = state["history"]
    if hist:
        last = max(hist)
        v = hist[last]
        start = cfg.start_cash
        f = state["followed"]
        out.append(f"\nValue {gbp(v['followed'])} · cash {gbp(f['cash'])} "
                   f"({f['cash'] / v['followed'] * 100:.0f}%)")
        out.append(f"Since {state['start_date']}: following calls {pct(v['followed'], start)} · "
                   f"your picks held {pct(v['held'], start)} · {cfg.benchmark} {pct(v['bench'], start)}")
        out.append(f"Fees {gbp(f['fees'])} · dividends {gbp(f['dividends'])} · "
                   f"cash interest {gbp(f.get('interest', 0.0))}")
    elif any(b.get("init") for b in state.get("queue", [])):
        out.append(f"\nStarting with {gbp(cfg.start_cash)} cash. First prices are recorded at today's close.")
    return "\n".join(out)


DASH_CSS = """
.tap{cursor:pointer;-webkit-tap-highlight-color:transparent}
.tap:hover .sym{color:#fff}
.tap:focus-visible{outline:1px solid var(--cobalt);outline-offset:2px;border-radius:6px}
#sheet{position:fixed;inset:0;display:none;z-index:10}
#sheet.open{display:block}
#sheet .back{position:absolute;inset:0;background:rgba(10,10,16,.72)}
#sheet .panel{position:absolute;left:0;right:0;bottom:0;max-height:92vh;overflow-y:auto;background:var(--bg);
border-radius:20px 20px 0 0;border-top:1px solid var(--line);padding:20px 16px 44px;max-width:760px;margin:0 auto}
.shead{display:flex;justify-content:space-between;align-items:flex-start;gap:12px}
.sname{font-size:22px;font-weight:500;line-height:1.2}
.x{background:var(--raised);color:var(--ivory);border:0;border-radius:40px;padding:8px 16px;font:inherit;
font-size:14px;cursor:pointer}
.sprice{font-size:34px;font-weight:500;line-height:1.1;margin:14px 0 4px}
.pos{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0 6px}
.pos .score{padding:12px}.pos .sval{font-size:17px}
.h3{font-size:15px;font-weight:500;margin:24px 0 4px}
.crow{display:grid;grid-template-columns:54px 1fr 76px 72px;gap:10px;align-items:center;padding:10px 0;
border-top:1px solid var(--line)}
.scrub{min-height:20px;font-size:13px;color:var(--ash);margin-top:12px}
.dchart{width:100%;height:auto;display:block;touch-action:pan-y}
.dchart text{fill:var(--slate);font-size:11px}
.key{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--ash);margin-top:8px}
.key i{display:inline-block;width:9px;height:9px;border-radius:9px;margin-right:6px;vertical-align:middle}
.key i.b{background:var(--cobalt)}.key i.s{border:2px solid var(--ivory)}
@media (max-width:520px){.pos{grid-template-columns:1fr 1fr}.crow{grid-template-columns:48px 1fr 70px 64px}}
@media (min-width:800px){#sheet .panel{top:5vh;bottom:auto;border-radius:20px;border:1px solid var(--line);
padding:28px 28px 36px}}
"""

DASH_JS = r"""
(function(){
var D=JSON.parse(document.getElementById('pp-data').textContent);
var sheet=document.getElementById('sheet'),body=document.getElementById('sheet-body');
var W=420,H=230,L=50,R=10,T=12,B=24;
function money(v){var big=Math.abs(v)>=1000;
 return '£'+v.toLocaleString('en-GB',{minimumFractionDigits:big?0:2,maximumFractionDigits:big?0:2});}
function fmtD(s){return new Date(s+'T00:00:00Z').toLocaleDateString('en-GB',
 {day:'numeric',month:'short',timeZone:'UTC'});}
function pct(a,b){if(!a||!b)return '<span class="muted">—</span>';var v=(a/b-1)*100;
 return '<span class="'+(v>=0?'up':'down')+'">'+(v>=0?'▲':'▼')+' '+Math.abs(v).toFixed(1)+'%</span>';}
var CH={buy:['Buy','chip-buy'],reduce:['Reduce','chip-sell'],sell:['Sell','chip-sell'],hold:['Watch','chip-hold']};
function chip(a){var c=CH[a]||[a,'chip-hold'];return '<span class="chip '+c[1]+'">'+c[0]+'</span>';}
function qty(q){return (+q.toPrecision(4)).toLocaleString('en-GB');}
function stat(l,v){return '<div class="score"><div class="cap">'+l+'</div><div class="sval">'+v+'</div></div>';}
function chart(s){
 var p=s.px;if(p.length<2)return '<p class="muted small">The chart appears after two trading days.</p>';
 var lo=Infinity,hi=-Infinity,i;
 for(i=0;i<p.length;i++){lo=Math.min(lo,p[i][1]);hi=Math.max(hi,p[i][1]);}
 var pad=(hi-lo)*0.14||hi*0.02||1;lo-=pad;hi+=pad;
 function x(k){return L+(W-L-R)*k/(p.length-1);}
 function y(v){return T+(H-T-B)*(1-(v-lo)/(hi-lo));}
 var o='<svg class="dchart" viewBox="0 0 '+W+' '+H+'" role="img" aria-label="Price since the start">';
 for(i=0;i<3;i++){var v=lo+(hi-lo)*(i+0.5)/3;
  o+='<line x1="'+L+'" x2="'+(W-R)+'" y1="'+y(v).toFixed(1)+'" y2="'+y(v).toFixed(1)+'" class="grid"/>';
  o+='<text x="'+(L-8)+'" y="'+(y(v)+4).toFixed(1)+'" text-anchor="end">'+money(v)+'</text>';}
 o+='<text x="'+L+'" y="'+(H-6)+'">'+fmtD(p[0][0])+'</text>';
 o+='<text x="'+(W-R)+'" y="'+(H-6)+'" text-anchor="end">'+fmtD(p[p.length-1][0])+'</text>';
 var pts='';for(i=0;i<p.length;i++)pts+=x(i).toFixed(1)+','+y(p[i][1]).toFixed(1)+' ';
 o+='<polygon points="'+L+','+(H-B)+' '+pts+(W-R)+','+(H-B)+'" fill="#5266eb" opacity="0.10"/>';
 o+='<polyline points="'+pts+'" fill="none" stroke="#ededf3" stroke-width="2" stroke-linejoin="round"/>';
 s.calls.forEach(function(c){
  if(c.a==='hold'||c.pending)return;var k=-1;
  for(var j=0;j<p.length;j++){if(p[j][0]<=c.d)k=j;}
  if(k<0)return;var cx=x(k).toFixed(1),cy=y(p[k][1]).toFixed(1);
  o+=c.a==='buy'?'<circle cx="'+cx+'" cy="'+cy+'" r="4.5" fill="#5266eb" stroke="#171721" stroke-width="1.5"/>'
   :'<circle cx="'+cx+'" cy="'+cy+'" r="4" fill="#171721" stroke="#ededf3" stroke-width="2"/>';});
 o+='<line class="sl" y1="'+T+'" y2="'+(H-B)+'" stroke="#70707d" stroke-width="1" visibility="hidden"/>';
 o+='<circle class="sc" r="4" fill="#ededf3" visibility="hidden"/></svg>';
 o+='<div class="key"><span><i class="b"></i>Buy call</span><span><i class="s"></i>Reduce or Sell call</span></div>';
 return o;}
function open(sym){
 var s=D[sym];if(!s)return;var p=s.px,last=p.length?p[p.length-1]:null,h='';
 h+='<div class="shead"><div><div class="cap">'+sym+'</div><div class="sname">'+(s.name||sym)+'</div></div>';
 h+='<button class="x" type="button" data-close="1">Close</button></div>';
 if(last){h+='<div class="sprice">'+money(last[1])+'</div><div class="small">'+pct(last[1],p[0][1]);
  h+=' <span class="muted">since '+fmtD(p[0][0])+' · close on '+fmtD(last[0])+'</span></div>';}
 else h+='<p class="muted">Prices appear after the first run settles.</p>';
 if(s.qty>0){h+='<div class="pos">'+stat('Shares',qty(s.qty))+stat('Value',money(s.value))
  +stat('Paid',money(s.cost))+stat('Gain',pct(s.value,s.cost))+'</div>';}
 else h+='<p class="muted small" style="margin:14px 0 0">Not held right now.</p>';
 h+='<div class="scrub" id="scrub">'+(p.length>1?'Touch the chart to see the price on any day.':'')+'</div>';
 h+=chart(s);
 h+='<div class="h3">Calls</div>';
 var cs=s.calls.slice().reverse().slice(0,30);
 if(!cs.length)h+='<p class="muted small">No calls recorded yet.</p>';
 cs.forEach(function(c){h+='<div class="crow"><div class="muted small">'+fmtD(c.d)+'</div><div>'+chip(c.a)
  +' <span class="muted small">'+(c.s==null?'':c.s)+'</span></div><div class="num small">'
  +(c.p?money(c.p):'<span class="muted">at close</span>')+'</div><div class="num small">'
  +(c.p&&last?pct(last[1],c.p):'')+'</div></div>';});
 h+='<div class="h3">Your trades</div>';
 if(!s.trades.length)h+='<p class="muted small">None yet.</p>';
 s.trades.slice().reverse().forEach(function(t){h+='<div class="crow"><div class="muted small">'+fmtD(t.d)
  +'</div><div class="small">'+(t.side==='BUY'?'Bought ':'Sold ')+qty(t.q)+'</div><div class="num small">'
  +money(t.v)+'</div><div class="num small muted">fee '+money(t.fee)+'</div></div>';});
 if(s.divs.length){h+='<div class="h3">Dividends</div>';
  s.divs.slice().reverse().forEach(function(d){h+='<div class="crow"><div class="muted small">'+fmtD(d.d)
   +'</div><div class="small">Paid to cash</div><div></div><div class="num small">'+money(d.amt)+'</div></div>';});}
 h+='<div class="foot">Prices are daily closes in pounds, updated after each run. Not live.</div>';
 body.innerHTML=h;sheet.classList.add('open');document.body.style.overflow='hidden';
 sheet.querySelector('.panel').scrollTop=0;
 var svg=body.querySelector('.dchart');
 if(svg){var sl=svg.querySelector('.sl'),sc=svg.querySelector('.sc'),out=document.getElementById('scrub');
  var lo=Infinity,hi=-Infinity;p.forEach(function(q){lo=Math.min(lo,q[1]);hi=Math.max(hi,q[1]);});
  var pad=(hi-lo)*0.14||hi*0.02||1;lo-=pad;hi+=pad;
  var mv=function(e){var r=svg.getBoundingClientRect(),fx=(e.clientX-r.left)/r.width*W;
   var k=Math.round((fx-L)/(W-L-R)*(p.length-1));k=Math.max(0,Math.min(p.length-1,k));
   var cx=L+(W-L-R)*k/(p.length-1),cy=T+(H-T-B)*(1-(p[k][1]-lo)/(hi-lo));
   sl.setAttribute('x1',cx);sl.setAttribute('x2',cx);sl.setAttribute('visibility','visible');
   sc.setAttribute('cx',cx);sc.setAttribute('cy',cy);sc.setAttribute('visibility','visible');
   out.textContent=fmtD(p[k][0])+' · '+money(p[k][1]);};
  svg.addEventListener('pointermove',mv);svg.addEventListener('pointerdown',mv);}
}
function close(){sheet.classList.remove('open');document.body.style.overflow='';}
document.addEventListener('click',function(e){
 if(e.target.closest('[data-close]')){close();return;}
 var el=e.target.closest('[data-sym]');if(el)open(el.getAttribute('data-sym'));});
document.addEventListener('keydown',function(e){
 if(e.key==='Escape')close();
 if(e.key==='Enter'&&e.target.getAttribute&&e.target.getAttribute('data-sym'))open(e.target.getAttribute('data-sym'));});
var hs=decodeURIComponent(location.hash.slice(1));if(D[hs])open(hs);
})();
"""


def _fmt_d(d: str) -> str:
    return date.fromisoformat(d).strftime("%-d %b %Y")


def _fmt_short(d: str) -> str:
    return date.fromisoformat(d).strftime("%-d %b")


def _chg(a: float, b: float) -> Optional[float]:
    return (a / b - 1) * 100 if b else None


def _pct_html(v: Optional[float]) -> str:
    if v is None:
        return "<span class='muted'>—</span>"
    cls = "up" if v >= 0 else "down"
    arrow = "▲" if v >= 0 else "▼"
    return f"<span class='{cls}'>{arrow} {abs(v):.1f}%</span>"


CHIP = {"buy": ("Buy", "chip-buy"), "reduce": ("Reduce", "chip-sell"), "sell": ("Sell", "chip-sell"),
        "hold": ("Watch", "chip-hold")}


def _chip(action: str, label: Optional[str] = None) -> str:
    text, cls = CHIP.get(action, (action.title(), "chip-hold"))
    return f"<span class='chip {cls}'>{html.escape(label or text)}</span>"


def svg_chart(history: Dict[str, dict], benchmark: str, start: float) -> str:
    dates = sorted(history)
    if len(dates) < 2:
        return "<p class='muted'>The chart appears after two trading days.</p>"
    series = [("Following calls", "followed", "#5266eb", ""),
              ("Your picks held", "held", "#ededf3", ""),
              (f"{benchmark} only", "bench", "#70707d", "5 5")]
    vals = [history[d][k] for d in dates for _, k, _, _ in series]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.12 or 40
    lo, hi = lo - pad, hi + pad
    W, H, L, R, T, B = 420, 260, 50, 10, 10, 26

    def x(i):
        return L + (W - L - R) * i / (len(dates) - 1)

    def y(v):
        return T + (H - T - B) * (1 - (v - lo) / (hi - lo))

    out = [f"<svg viewBox='0 0 {W} {H}' class='chart' role='img' aria-label='Portfolio value over time'>"]
    for k in range(4):
        v = lo + (hi - lo) * (k + 0.5) / 4
        out.append(f"<line x1='{L}' x2='{W - R}' y1='{y(v):.1f}' y2='{y(v):.1f}' class='grid'/>"
                   f"<text x='{L - 8}' y='{y(v) + 4:.1f}' text-anchor='end'>£{v:,.0f}</text>")
    if lo < start < hi:
        out.append(f"<line x1='{L}' x2='{W - R}' y1='{y(start):.1f}' y2='{y(start):.1f}' class='base'/>")
    out.append(f"<text x='{L}' y='{H - 8}'>{_fmt_short(dates[0])}</text>"
               f"<text x='{W - R}' y='{H - 8}' text-anchor='end'>{_fmt_short(dates[-1])}</text>")
    f_pts = [(x(i), y(history[d]["followed"])) for i, d in enumerate(dates)]
    area = " ".join(f"{a:.1f},{b:.1f}" for a, b in f_pts)
    out.append(f"<polygon points='{L},{H - B} {area} {W - R},{H - B}' fill='#5266eb' opacity='0.10'/>")
    for _, k, col, dash in reversed(series):
        pts = " ".join(f"{x(i):.1f},{y(history[d][k]):.1f}" for i, d in enumerate(dates))
        da = f" stroke-dasharray='{dash}'" if dash else ""
        out.append(f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='2.4'"
                   f" stroke-linejoin='round'{da}/>")
    lx, ly = f_pts[-1]
    out.append(f"<circle cx='{lx:.1f}' cy='{ly:.1f}' r='4' fill='#5266eb'/></svg>")
    legend = "".join(f"<span><i style='background:{c}'></i>{html.escape(n)}</span>" for n, _, c, _ in series)
    return "".join(out) + f"<div class='legend'>{legend}</div>"


def sparkline(points: List[float]) -> str:
    if len(points) < 2:
        return "<svg class='spark' viewBox='0 0 120 32'></svg>"
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1
    pts = " ".join(f"{120 * i / (len(points) - 1):.1f},{28 - 24 * (p - lo) / span:.1f}"
                   for i, p in enumerate(points))
    col = "#ededf3" if points[-1] >= points[0] else "#70707d"
    return (f"<svg class='spark' viewBox='0 0 120 32' preserveAspectRatio='none'>"
            f"<polyline points='{pts}' fill='none' stroke='{col}' stroke-width='1.6'/></svg>")


def call_scores(state, horizon: int = 5) -> Dict[str, dict]:
    """How often actionable calls were right `horizon` trading days later."""
    out = {"buy": {"n": 0, "right": 0, "moves": []}, "exit": {"n": 0, "right": 0, "moves": []},
           "pending": 0}
    for c in state.get("calls", []):
        if c["action"] == "hold" or not c.get("price"):
            continue
        closes = state.get("closes", {}).get(c["symbol"], {})
        later = sorted(d for d in closes if d > c["date"])
        if len(later) < horizon:
            out["pending"] += 1
            continue
        move = _chg(closes[later[horizon - 1]], c["price"])
        bucket = out["buy"] if c["action"] == "buy" else out["exit"]
        bucket["n"] += 1
        bucket["moves"].append(move)
        if (move > 0) == (c["action"] == "buy"):
            bucket["right"] += 1
    return out


def dashboard_html(state, cfg: Config) -> str:
    hist = state["history"]
    start = cfg.start_cash
    last = max(hist) if hist else None
    v = hist[last] if last else {"followed": start, "held": start, "bench": start}
    f = state["followed"]
    prices = state.get("last_prices", {})
    closes = state.get("closes", {})
    calls = state.get("calls", [])
    pend = (state.get("queue") or [{}])[-1]
    latest_call: Dict[str, dict] = {c["symbol"]: c for c in calls}
    for c in pend.get("calls", []):
        latest_call[c["symbol"]] = {**c, "date": pend["signal_date"], "pending": True}

    ret_f = _chg(v["followed"], start) or 0.0
    ret_b = _chg(v["bench"], start) or 0.0
    gap = ret_f - ret_b
    gap_txt = f"{'+' if gap >= 0 else '−'}{abs(gap):.1f} pts vs {cfg.benchmark}"

    def mini(label, val):
        return (f"<div class='mini'><div class='cap'>{html.escape(label)}</div>"
                f"<div class='mval'>{gbp(val)}</div>{_pct_html(_chg(val, start))}</div>")

    # Scorecard
    sc = call_scores(state)

    def score_block(title, b, verb):
        if not b["n"]:
            return (f"<div class='score'><div class='cap'>{title}</div><div class='sval'>—</div>"
                    f"<div class='muted small'>None scored yet</div></div>")
        rate = b["right"] / b["n"] * 100
        avg = sum(b["moves"]) / len(b["moves"])
        return (f"<div class='score'><div class='cap'>{title}</div><div class='sval'>{rate:.0f}%</div>"
                f"<div class='muted small'>{b['right']} of {b['n']} {verb} · avg move {avg:+.1f}%</div></div>")

    scorecard = (score_block("Buy calls right", sc["buy"], "rose")
                 + score_block("Reduce / Sell calls right", sc["exit"], "fell"))
    score_note = (f"A call is scored 5 trading days after it's made. "
                  f"{sc['pending']} call{'s' if sc['pending'] != 1 else ''} still waiting.")

    # Holdings
    hold_rows = []
    for sym, q in sorted(f["holdings"].items(), key=lambda kv: -kv[1] * prices.get(kv[0], 0)):
        val = q * prices.get(sym, 0)
        w = val / v["followed"] * 100 if v["followed"] else 0
        lc = latest_call.get(sym)
        hold_rows.append(
            f"<div class='row tap' data-sym='{sym}' role='button' tabindex='0'>"
            f"<div class='sym'>{sym}<div class='muted small'>{q:,.4g} shares</div></div>"
            f"<div class='bar'><span style='width:{min(w * 2.5, 100):.0f}%'></span>"
            f"<div class='muted small'>{w:.1f}% of portfolio</div></div>"
            f"<div class='num'>{gbp(val)}<div class='small'>{_pct_html(_chg(val, f['cost'].get(sym, 0)))}</div></div>"
            f"<div class='chipcol'>{_chip(lc['action']) if lc else ''}</div></div>")
    cash_w = f["cash"] / v["followed"] * 100 if v["followed"] else 100
    hold_rows.append(
        f"<div class='row'><div class='sym'>Cash<div class='muted small'>earning "
        f"{cfg.cash_interest * 100:.2f}%</div></div><div class='bar'><span class='cash' "
        f"style='width:{min(cash_w, 100):.0f}%'></span><div class='muted small'>{cash_w:.1f}% of portfolio</div>"
        f"</div><div class='num'>{gbp(f['cash'])}</div><div class='chipcol'></div></div>")

    # Watchlist
    universe = (pend.get("universe") or sorted(closes))
    watch_rows = []
    for sym in universe:
        series = [closes.get(sym, {})[d] for d in sorted(closes.get(sym, {}))]
        lc = latest_call.get(sym)
        chip = _chip(lc["action"], f"{CHIP.get(lc['action'], ('', ''))[0]} {lc['score'] if lc['score'] is not None else ''}".strip()) if lc else ""
        held = "<span class='dot' title='Held'></span>" if sym in f["holdings"] else ""
        watch_rows.append(
            f"<div class='wrow tap' data-sym='{sym}' role='button' tabindex='0'>"
            f"<div class='sym'>{held}{sym}</div>{sparkline(series)}"
            f"<div class='num small'>{_pct_html(_chg(series[-1], series[0]) if len(series) > 1 else None)}</div>"
            f"<div class='chipcol'>{chip}</div></div>")

    # Calls timeline (actionable only)
    timeline = []
    for c in reversed([c for c in calls if c["action"] != "hold"][-25:]):
        now_px = prices.get(c["symbol"])
        move = _chg(now_px, c["price"]) if now_px and c.get("price") else None
        timeline.append(
            f"<div class='trow tap' data-sym='{c['symbol']}' role='button' tabindex='0'>"
            f"<div class='muted small'>{_fmt_short(c['date'])}</div>"
            f"<div class='sym'>{c['symbol']}</div><div>{_chip(c['action'])}"
            f" <span class='muted small'>{c['score'] if c['score'] is not None else ''}</span></div>"
            f"<div class='num small'>{gbp(c['price'])}</div><div class='num small'>{_pct_html(move)}</div></div>")
    timeline_html = "".join(timeline) or "<p class='muted'>No Buy, Reduce or Sell calls yet.</p>"

    trades = "".join(
        f"<div class='trow tap' data-sym='{t['symbol']}' role='button' tabindex='0'>"
        f"<div class='muted small'>{_fmt_short(t['date'])}</div><div class='sym'>{t['symbol']}</div>"
        f"<div class='small'>{'Bought' if t['side'] == 'BUY' else 'Sold'} {t['qty']:,.4g}</div>"
        f"<div class='num small'>{gbp(t['value'])}</div><div class='num small muted'>fee {gbp(t['fee'])}</div></div>"
        for t in reversed(state["trades"][-20:])) or "<p class='muted'>No trades yet.</p>"

    # Per-stock detail shown when a row is tapped.
    meta = state.get("meta", {})
    detail = {}
    for sym in dict.fromkeys(list(universe) + list(f["holdings"]) + [c["symbol"] for c in calls]):
        sym_calls = [{"d": c["date"], "a": c["action"], "s": c["score"], "p": c.get("price")}
                     for c in calls if c["symbol"] == sym]
        sym_calls += [{"d": pend["signal_date"], "a": c["action"], "s": c["score"], "p": None, "pending": True}
                      for c in pend.get("calls", []) if c["symbol"] == sym]
        q = f["holdings"].get(sym, 0.0)
        detail[sym] = {
            "name": (meta.get(sym) or {}).get("name") or "",
            "px": [[d, closes[sym][d]] for d in sorted(closes.get(sym, {}))],
            "calls": sym_calls,
            "trades": [{"d": t["date"], "side": t["side"], "q": t["qty"], "v": t["value"], "fee": t["fee"]}
                       for t in state["trades"] if t["symbol"] == sym],
            "divs": [{"d": x["date"], "amt": x["amount"]} for x in state.get("dividend_log", [])
                     if x["symbol"] == sym],
            "qty": q, "value": round(q * prices.get(sym, 0.0), 2), "cost": round(f["cost"].get(sym, 0.0), 2),
        }
    payload = html.escape(json.dumps(detail, separators=(",", ":")), quote=False)

    started = _fmt_d(state["start_date"]) if state.get("start_date") else "today"
    return f"""<!DOCTYPE html><html lang="en-GB"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Paper portfolio</title><style>
:root{{--bg:#171721;--card:#1e1e2a;--raised:#272735;--line:rgba(226,227,237,.08);--slate:#70707d;--ash:#c3c3cc;
--ivory:#ededf3;--cobalt:#5266eb;--up:#a9d8bd;--down:#e9a8a8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ivory);
font:400 16px/1.5 Inter,-apple-system,"Segoe UI",system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:760px;margin:0 auto;padding:28px 16px 56px}}
.cap{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--ash)}}
.muted{{color:var(--ash)}}.small{{font-size:13px;line-height:1.35}}
h1{{font-weight:500;font-size:44px;letter-spacing:.01em;line-height:1.1;margin:10px 0 6px}}
h2{{font-weight:500;font-size:19px;letter-spacing:.01em;margin:0 0 4px}}
.lede{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:4px}}
.pill{{border:1px solid var(--line);border-radius:40px;padding:4px 12px;font-size:13px;color:var(--ash)}}
.pill.lead{{border-color:var(--cobalt);color:var(--ivory)}}
.up{{color:var(--up)}}.down{{color:var(--down)}}
.card{{background:var(--card);border-radius:12px;padding:22px;margin-top:16px}}
.card .sub{{color:var(--ash);font-size:13px;margin-bottom:14px}}
.minis{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:22px}}
.mini{{background:var(--card);border-radius:12px;padding:14px}}.mval{{font-size:20px;font-weight:500;margin:4px 0 2px}}
.mini .cap{{font-size:11px}}
.chart{{width:100%;height:auto;display:block}}.chart text{{fill:var(--slate);font-size:11px}}
.grid{{stroke:var(--line)}}.base{{stroke:var(--slate);stroke-width:1;opacity:.35;stroke-dasharray:2 4}}
.legend{{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ash);margin-top:10px}}
.legend i{{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.scores{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.score{{background:var(--raised);border-radius:12px;padding:14px}}.sval{{font-size:30px;font-weight:500;margin:2px 0}}
.row,.wrow,.trow{{display:grid;align-items:center;gap:12px;padding:12px 0;border-top:1px solid var(--line)}}
.row{{grid-template-columns:1.1fr 1.4fr 1fr 70px}}.wrow{{grid-template-columns:1fr 1.3fr 70px 92px}}
.trow{{grid-template-columns:52px 1fr 1.1fr 72px 64px}}
.list>div:first-child{{border-top:0}}
.sym{{font-weight:500;font-size:15px}}.num{{text-align:right;font-variant-numeric:tabular-nums}}
.bar span{{display:block;height:6px;border-radius:6px;background:var(--cobalt);margin-bottom:4px}}
.bar span.cash{{background:var(--slate)}}
.spark{{width:100%;height:30px}}.chipcol{{text-align:right}}
.chip{{display:inline-block;border-radius:40px;padding:2px 10px;font-size:12px;white-space:nowrap}}
.chip-buy{{background:rgba(82,102,235,.18);color:#c9d0ff}}
.chip-sell{{border:1px solid var(--slate);color:var(--ivory)}}.chip-hold{{color:var(--slate);border:1px solid var(--line)}}
.dot{{display:inline-block;width:6px;height:6px;border-radius:6px;background:var(--cobalt);margin-right:6px;vertical-align:middle}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}.stats .sval{{font-size:18px}}
.foot{{color:var(--slate);font-size:12px;margin-top:22px;line-height:1.6}}
@media (max-width:520px){{h1{{font-size:38px}}.minis{{grid-template-columns:1fr 1fr}}.minis .mini:first-child{{grid-column:1/-1}}
.stats{{grid-template-columns:1fr 1fr}}.row{{grid-template-columns:1fr 1fr 1fr}}.row .chipcol{{display:none}}
.trow{{grid-template-columns:44px 1fr 1fr 60px}}.trow>:nth-child(4){{display:none}}
.wrow{{grid-template-columns:1fr 1fr 64px 84px}}.card{{padding:18px}}}}
{DASH_CSS}</style></head><body><div class="wrap">
<div class="cap">Paper portfolio · demo money</div>
<h1>{gbp(v['followed'])}</h1>
<div class="lede">{_pct_html(ret_f)}<span class="muted small">since {started}</span>
<span class="pill{' lead' if gap >= 0 else ''}">{gap_txt}</span></div>
<div class="minis">{mini('Following calls', v['followed'])}{mini('Your picks held', v['held'])}{mini(cfg.benchmark + ' only', v['bench'])}</div>

<div class="card"><h2>Value over time</h2><div class="sub">All three started with {gbp(start)}. Prices to {_fmt_d(last) if last else '—'}.</div>
{svg_chart(hist, cfg.benchmark, start)}</div>

<div class="card"><h2>How good are the calls?</h2><div class="sub">{score_note}</div>
<div class="scores">{scorecard}</div></div>

<div class="card"><h2>Holdings</h2><div class="sub">What the portfolio holds now · tap a stock for its chart, calls and trades</div>
<div class='list'>{''.join(hold_rows)}</div></div>

<div class="card"><h2>Your stocks</h2><div class="sub">Price since the start · latest call and score · <span class="dot"></span>held now</div>
<div class='list'>{''.join(watch_rows)}</div></div>

<div class="card"><h2>Calls and what happened next</h2><div class="sub">Price when called · change since</div>
<div class='list'>{timeline_html}</div></div>

<div class="card"><h2>Trades</h2><div class="sub">Filled at each market's first close after the call</div><div class='list'>{trades}</div></div>

<div class="card"><h2>Costs and income</h2><div class="stats" style="margin-top:12px">
<div class="score"><div class="cap">Fees</div><div class="sval">{gbp(f['fees'])}</div></div>
<div class="score"><div class="cap">Dividends</div><div class="sval">{gbp(f['dividends'])}</div></div>
<div class="score"><div class="cap">Cash interest</div><div class="sval">{gbp(f.get('interest', 0.0))}</div></div>
<div class="score"><div class="cap">Cash</div><div class="sval">{gbp(f['cash'])}</div></div></div></div>

<div class="foot">Simulated with demo money using {html.escape(cfg.platform)} costs: {cfg.fx_fee * 100:.2f}% FX fee on non-GBP trades,
{cfg.stamp_duty * 100:.1f}% stamp duty on UK share purchases, {cfg.cash_interest * 100:.2f}% interest on cash,
US dividends net of {cfg.us_withholding * 100:.0f}% withholding. Not financial advice.</div>
</div>
<div id="sheet" aria-hidden="true"><div class="back" data-close="1"></div>
<div class="panel" role="dialog" aria-label="Stock detail"><div id="sheet-body"></div></div></div>
<script type="application/json" id="pp-data">{payload}</script>
<script>{DASH_JS}</script>
</body></html>"""


def send_dashboard(path: Path, caption: str) -> None:
    import requests

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        return
    data = {"chat_id": chat, "caption": caption}
    thread = os.getenv("TELEGRAM_MESSAGE_THREAD_ID", "").strip()
    if thread:
        data["message_thread_id"] = thread
    with open(path, "rb") as fh:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendDocument", data=data,
                          files={"document": (path.name, fh, "text/html")}, timeout=30)
    if r.status_code >= 400:
        logger.warning("Dashboard send failed: HTTP %s", r.status_code)


def notify(message: str) -> None:
    try:
        from src.notification import NotificationService

        svc = NotificationService()
        if svc.is_available():
            svc.send(message)
    except Exception as e:
        logger.warning("Notification failed: %s", type(e).__name__)


def dashboard_due(today: str) -> bool:
    mode = os.getenv("PAPER_DASHBOARD", "").strip().lower() or "weekly"
    if mode == "never":
        return False
    if mode == "always" or os.getenv("GITHUB_EVENT_NAME") == "workflow_dispatch":
        return True
    return date.fromisoformat(today).weekday() == 4  # Friday


# --------------------------------------------------------------------------

def run(cfg: Config, md: MarketData, signals: List[Signal], today: str,
        now: Optional[datetime] = None) -> Tuple[dict, str]:
    now = now or datetime.combine(date.fromisoformat(today), time(18, 30), timezone.utc)
    if cfg.ledger_path.exists():
        state = json.loads(cfg.ledger_path.read_text())
    else:
        state = init_state(cfg, today)
    queue = state.setdefault("queue", [])
    if state.get("pending"):  # ledgers saved before the queue existed
        queue.append(state.pop("pending"))
        queue[-1].setdefault("signal_ts", datetime.combine(
            date.fromisoformat(queue[-1]["signal_date"]), time(10), timezone.utc).isoformat())

    usable = set(ensure_meta(state, md, list(dict.fromkeys(
        (cfg.universe or []) + [s.symbol for s in signals] + [cfg.benchmark]
        + list(state["followed"]["holdings"])))))
    signals = [s for s in signals if s.symbol in usable]

    trades: List[dict] = []
    fill_date, status = None, ("waiting" if queue else "none")
    while queue:
        t, fd, st = settle(state, cfg, md, queue[0], now)
        if st != "filled":
            break
        queue.pop(0)
        trades += t
        fill_date, status = fd, "filled"
    mark_to_market(state, cfg, md, today)

    queued: List[Signal] = []
    if signals:
        queued = queue_orders(state, cfg, signals, today, now)

    cfg.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg.ledger_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(cfg.ledger_path)

    logger.info("Ledger updated: %d trade(s) filled, %d order(s) queued", len(trades), len(queued))
    return state, summary(state, cfg, trades, fill_date, status, queued, signals)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    cfg = Config.from_env()
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()
    signals = load_signals(cfg.lookback_hours, set(cfg.universe or []) | {cfg.benchmark} if cfg.universe else None)
    logger.info("%d analysis call(s) found", len(signals))
    try:
        state, message = run(cfg, MarketData(), signals, today, now)
    except Exception as e:
        logger.error("Paper trading failed: %s", e)
        notify(f"📒 Paper portfolio update failed: {e}")
        return 1
    notify(message)
    if dashboard_due(today) and state["history"]:
        import tempfile

        path = Path(tempfile.gettempdir()) / f"paper-portfolio-{today}.html"
        path.write_text(dashboard_html(state, cfg))
        send_dashboard(path, "Paper portfolio dashboard — open in your browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
