# Live Auction Playbook

Use this during bidding. Keep runs short, consistent, and state-driven.

## 1) Set The Run Mode

- `Objective`:
  - `expected_gross`: best default for stable value capture.
  - `win_probability`: use when you want opponent-aware portfolio choices.
- `Budget basis`: use `personal` in live drafting (your real remaining bankroll).
- `Optimizer cost`:
  - `target_bid`: execution-first (what you can pay efficiently).
  - `market_fair_bid`: pressure test (what survives competitive pricing).

Practical default: `expected_gross + personal + market_fair_bid` for scan, then confirm bids with `target_bid`.

## 2) Use Previous Bid Correctly

- `Previous bid` is for the **current movie only**.
- Set it to current high bid in the room.
- Integer mode enforces next legal bid as `previous_bid + 1`.
- Do not carry a previous bid from one movie to the next.

## 3) Read The Key Columns Fast

- `TgtInt`: your personal value anchor.
- `TgtMktInt`: market-pressure anchor.
- `MaxInt`: hard stop.

Action rule:
- Bid confidently below `TgtInt`.
- Between `TgtInt` and `TgtMktInt`: only continue if movie is core and portfolio still robust.
- Never exceed `MaxInt`.

## 4) State Changes After Every Result

After each movie ends:

1. Assign winner and final price in draft state.
2. Confirm winner budget is decremented.
3. Confirm movie is removed from available pool.
4. Set next `current movie` and its `previous bid`.
5. Rerun dashboard immediately.

If you skip reruns, recommendations become stale.

## 5) Two-Pass Live Workflow

For each movie:

1. Run with `market_fair_bid` to test if it remains attractive under pressure.
2. Run/interpret with `target_bid` for final executable decision.

If a movie fails both passes, pivot to alternates and preserve budget.
