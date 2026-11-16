# Paper-trading ledger (demo money)

After each daily analysis, `scripts/paper_trade.py` follows the report's calls
with demo money and sends a summary to your notification channel. It keeps
three portfolios in GBP, all starting from the same cash:

| Portfolio | What it does |
| --- | --- |
| Following calls | Starts in cash and only trades on the analyser's calls |
| Your picks held | Buys your stock list on day one and never trades |
| Benchmark only | Everything in `PAPER_BENCHMARK` (default `VUAG.L`) |

*Your picks held* vs *Benchmark* shows whether your stock choices beat the
index. *Following calls* vs *Your picks held* shows whether the calls helped.

## Setup

1. Make up a long random password and save it as a repository secret named
   `PAPER_LEDGER_KEY` (repo → Settings → Secrets and variables → Actions).
   Keep a copy: without it the ledger can't be read, and changing it stops
   the step rather than starting over.
2. Optional repository variables: `PAPER_START_CASH` (default 10000),
   `PAPER_WEIGHTS` such as `PLTR:0.5,RKLB:0.5,CRWD:0.5` (default weight 1).

The ledger is saved encrypted as the only file on the `paper-ledger` branch,
so it stays private on a public repository. To start again from scratch,
delete that branch.

## Trading rules (Following calls)

A "slot" is the portfolio's value × the stock's weight ÷ the total weight.

| Report call | Trade |
| --- | --- |
| Buy / Strong Buy / Add | Top the holding up to one slot, using available cash |
| Watch / Hold | Nothing |
| Reduce | Sell half |
| Sell | Sell everything |

When cash is short, the highest-scoring Buy calls are filled first.

## Realism

Defaults model a UK investor using Trading 212 in a stocks and shares ISA:

- Orders fill at the close on the day of the call, and are settled on the
  next run, so no later prices are used.
- 0.15% FX fee on buying and selling non-GBP stocks (`PAPER_FX_FEE`).
- 0.5% stamp duty on buying UK shares, not ETFs (`PAPER_STAMP_DUTY`).
- Fractional shares. Set `PAPER_UK_FRACTIONAL=false` for whole UK shares.
- Interest on uninvested cash, added daily (`PAPER_CASH_INTEREST`, default
  3.55% a year). The real rate is variable, so update it when it changes.
- Dividends paid in cash on the ex-date. US dividends net of 15% withholding
  (`PAPER_US_WITHHOLDING`, the W-8BEN rate); no UK tax inside an ISA.
- Prices, dividends and exchange rates come from Yahoo Finance.

For Freetrade's free plan instead: `PAPER_PLATFORM=Freetrade Basic`,
`PAPER_FX_FEE=0.0099`, `PAPER_UK_FRACTIONAL=false`, `PAPER_CASH_INTEREST=0`.

## What you receive

- A short summary after every run: trades filled, new orders, value and
  returns for all three portfolios, fees, dividends and interest.
- A dashboard file (HTML, opens in your phone's browser) every Friday and
  after manual runs: value over time, how often the calls were right five
  trading days later, holdings, each stock's price and latest call, and what
  happened after each call. Tap any stock for its own chart, call history,
  trades and dividends. `PAPER_DASHBOARD` = `weekly` (default), `always`
  or `never`.

Nothing about holdings or values is written to the public Actions log.

## Config

| Name | Where | Default |
| --- | --- | --- |
| `PAPER_LEDGER_KEY` | Actions secret | unset (step skipped) |
| `PAPER_START_CASH` | Actions variable | `10000` |
| `PAPER_WEIGHTS` | Actions variable | all `1` |
| `PAPER_BENCHMARK` | Actions variable | `VUAG.L` |
| `PAPER_FX_FEE` | Actions variable | `0.0015` |
| `PAPER_CASH_INTEREST` | Actions variable | `0.0355` |
| `PAPER_UK_FRACTIONAL` | Actions variable | `true` |
| `PAPER_PLATFORM` | Actions variable | `Trading 212` |
| `PAPER_DASHBOARD` | Actions variable | `weekly` |
| `PAPER_STAMP_DUTY`, `PAPER_US_WITHHOLDING`, `PAPER_LOOKBACK_HOURS` | env | `0.005`, `0.15`, `12` |
