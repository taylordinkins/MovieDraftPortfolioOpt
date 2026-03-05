# Live Auction Playbook

Use this document during bidding. Keep runs short, consistent, and state-driven.

---

## 1) Operating Mode — Set Once Per Session

**Objective**

- `win_probability` is your primary operating mode. Your goal is to finish first in total domestic gross — this is a rank-order competition, not a value-capture exercise. The win_probability optimizer accounts for simulated opponent portfolios and maximizes P(you > all others), which is theoretically the correct objective for a first-place-wins contest.
- `expected_gross` is a fast diagnostic and sanity check. Use it when time is tight or you want a quick ranking pass before switching back to win_probability. It does not account for opponent behavior or relative standing.

**Budget basis**: always `personal`. Your executable remaining bankroll is the correct constraint — league basis does not reflect your actual bidding ceiling.

**Optimizer cost column**

- `market_fair_bid`: primary operating basis. This prices each movie as a proportional share of the total remaining league budget, making costs realistic relative to what you'd actually pay in a competitive room. The stresstest cap (0.75 × personal budget = $150) ensures all movies including Avengers are eligible under this mode.
- `target_bid`: secondary confirmation pass. This prices movies as a proportional share of *your* remaining budget, adjusted for risk and the 0.95 bid multiplier. It produces much smaller numbers because it is anchored to your personal bankroll fraction, not the league's. Use it to confirm your executable intention, not for portfolio selection.
- `current_price`: diagnostic only. Do not use for optimizer selection or bid decisions.

**Standard session workflow**: load `realistic_params` → run `win_probability + personal + market_fair_bid` → read portfolio → confirm bids with `target_bid` column.

---

## 2) Per-Movie Decision Loop — Fast Read

### Step A: Set draft context before every movie

1. Set **Current Movie** to the active ticker.
2. Set **Previous Bid** to the current high bid in the room. Integer mode enforces your minimum legal next bid as `previous_bid + 1`. Reset to 0 for each new movie — never carry a previous bid across movies.
3. Verify your remaining budget in people.csv reflects actual spend to date.

### Step B: Read the four action columns

| Column | What it represents | Actionable meaning |
|--------|-------------------|-------------------|
| `MinBid` | `previous_bid + 1` | Legal floor — you cannot bid below this |
| `TgtInt` | Your proportional budget share of this movie, risk-adjusted, at 0.95× multiplier | Comfortable execution zone — bid confidently here |
| `TgtMktInt` | The same share scaled to the full league's remaining budget, risk-adjusted | Market-pressure reference — the price a competitive field would pay |
| `MaxInt` | `min(market_fair_bid, personal_remaining)` — hard ceiling | Walk-away price — never exceed this |

**Mathematical grounding**: `TgtInt = (AdjExp / pool_total_AdjExp) × personal_remaining × (1 − risk_penalty) × 0.95`. The 0.95 bid multiplier reflects your patient-bidder discount. `MaxInt` is capped at `market_fair_bid` because paying more than the league's proportional allocation for a movie is value-destructive regardless of your budget.

### Step C: Three-zone bidding rule

- **Below TgtInt**: bid freely. You are within your proportional value with margin.
- **TgtInt to TgtMktInt**: continue only if (a) this movie is in the optimizer's selected set AND (b) win probability remains above baseline (>12.5% in an 8-player league). You are paying more than your personal anchor but still within competitive market value.
- **Above TgtMktInt to MaxInt**: reserve for genuine must-haves where (a) no good alternates remain and (b) remaining budget can still build a competitive portfolio. The strategy guide's maximin rule applies: a movie that destroys your win probability under the aggressive-bidup profile is not a must-have.
- **Above MaxInt**: stop. This is the absolute walk-away point, period. Paying above market_fair_bid for any movie is negative expected value in the aggregate.

### Step D: When TgtInt < MinBid

This happens when the bidding has already exceeded your proportional value anchor. It is a signal, not a command to stop bidding. Check:
1. Is this movie in the optimizer's selected portfolio? If yes, you may still have room up to MaxInt.
2. What does `Score` and `P(Edge)` show? If P(Edge) is above 0.50 and Score is above median, the model still sees the movie as attractive at the current price.
3. If both are weak, pivot to the alternates table.

---

## 3) State Updates After Every Result

After each movie closes:

1. **Assign** winner and final price in Draft State. This removes the movie from the available pool.
2. **Decrement winner's budget** by the final price — do this for *every* player, not just yourself. The model uses all remaining budgets to compute `market_fair_bid` (league_remaining × gross_share). Stale opponent budgets will cause stale bid guidance.
3. **Set next current movie** and reset Previous Bid to 0 (or the opening bid if announced in advance).
4. **Rerun dashboard**. After any anchor-tier movie closes, a rerun is mandatory: the remaining pool value, league budget, and proportional shares all change, repricing every remaining movie. For cheap filler movies where the outcome will not change your strategy, a rerun can be deferred.

**Why opponent budgets matter**: Sequential auction theory (Kobayashi, 2023) shows that budget-constrained bidders become less competitive in later auctions as their remaining capital falls. An opponent who spent $150 on Mario in Cycle 1 has only $50 left for everything in Cycles 2–3. Tracking this gives you a real competitive advantage: you know when you are facing depleted opponents and can acquire movies at below-market prices.

---

## 4) Sequential Budget Management Across Cycles

The draft spans three cycles of roughly 14 movies each. Budget allocation across cycles is mathematically the most important strategic decision you will make.

### The core tradeoff

Spending budget in Cycle 1 earns you a movie now but reduces your bidding power in Cycles 2 and 3. The option value of holding budget is highest when:
- High-value movies are concentrated in later cycles (Cycle 2: Toy Story $83, Minions $75, Spider-Man $95; Cycle 3: Avengers $141 — all in market_fair_bid terms)
- Opponents are likely to overspend early (empirical research on fantasy auction drafts confirms early-round overbidding is the most common observed pattern)

### Budget checkpoints by cycle

Use these rough targets to monitor your position:

| After Cycle 1 (~14 movies, Mar–Jun) | Target remaining budget |
|--------------------------------------|------------------------|
| Winning nothing in Cycle 1 | ~$200 (full) — maximum flexibility |
| Winning 1–2 mid-tier films | $120–160 remaining |
| Winning a major Cycle 1 anchor (e.g. Mario $115, Mandalorian $51) | $85–140 remaining |

Entering Cycle 2 with less than $80 materially constrains your ability to compete for Spider-Man, Toy Story, or Minions. Entering Cycle 3 with less than $50 leaves you unable to meaningfully bid on Avengers ($141 market-fair) or Dune ($61).

### The enforcer cost

You identified the "enforcer dilemma": bidding up movies you don't want to prevent opponents getting them cheaply. This has a real cost. The rule: **enforce only if you have budget to spare beyond your remaining target for the cycle**. If enforcing a movie would drop you below the cycle target above, let it go. One opponent winning a cheap mid-tier film is less damaging than you losing your ability to compete for a blockbuster.

### Signals to spend aggressively NOW

- The optimizer's Opt column marks the movie as selected in your portfolio
- `break_even_avail` (in pool stats) is below the current bid level — meaning the movie is cheap relative to remaining available pool
- Opponents are visibly running low on budget (reduces competition for remaining movies)
- Only 1–2 remaining big movies have better expected value than this one

### Signals to hold budget

- Multiple higher-value movies remain in later cycles (confirmed by the optimizer selecting them over this one)
- Win probability under the `aggressive_bidup` profile is still healthy without this movie
- The alternates table shows multiple viable substitutes at lower cost

---

## 5) Interpreting the Monte Carlo Summary

Run the full Monte Carlo summary after each major state change. The key metrics and what they tell you:

### Win probability
The simulation runs 10,000 scenarios across 8 opponents and counts how often your portfolio total gross exceeds all others. With n=10,000 and p≈0.20, the standard error is approximately ±0.4 percentage points, so differences less than ~2pp are statistical noise. Compare against the random baseline of 12.5% (1/8). A result of 15–18% is modestly above baseline; 22%+ is strong.

When win probability under `balanced_field` looks healthy but drops sharply under `aggressive_bidup`, you have fragile concentration — your portfolio depends on blockbusters not getting bid up. This is the maximin signal to diversify or lower aggression.

### P(gross_equiv < spend)
This is P(your portfolio's actual domestic gross, in auction-dollar equivalent terms, falls below what you paid). It is not the same as losing the draft — you can still win even if this is positive. But values above 0.30 indicate the portfolio is unlikely to justify its cost in gross terms, which is a signal to increase selectivity or lower bids.

### Concentration downside
P(your portfolio underperforms AND a single movie drove most of it). A low value here means your downside is diversified — good. A high value means a single bad movie could sink you. For your win condition (total gross), some concentration in blockbusters is mathematically correct: Avengers alone at $540M expected HSX is worth more than five mid-tier films combined. However, concentration downside above 0.20 in the `t_copula` mode is a warning that your tail risk is correlated — bad box office seasons hit multiple movies simultaneously, as the t-copula models with its symmetric fat-tail dependence.

### Aggression sensitivity [0.9×, 1.0×, 1.1×]
This shows how your expected surplus and win probability change if actual competitive prices are 10% lower or higher than estimated. If the 1.1× scenario shows a large drop in expected surplus or positive-surplus probability, you have thin margin for price error — back off on any movies near TgtMktInt. If the 0.9× scenario shows a big improvement, market underbidding (which you've observed as likely for mid-tier movies) is particularly valuable for your portfolio.

### Total AdjExp (Selected Portfolio)
The new total displayed under "Selected Portfolio" in the GUI shows the sum of adjusted expected values across your selected movies. Compare this to the league baseline: at perfectly proportional bidding, each of 8 players would get approximately 1/8 of the total pool AdjExp. With current pool total HSX ~$6,140 and 8 players, the fair-share baseline is ~$768 in expected domestic gross per player. To reliably win the draft, your selected portfolio total AdjExp should exceed $900–1000, ideally with reasonable win probability across all three opponent profiles.

---

## 6) Reading the Alternates Table

The alternates table shows movies NOT in the optimizer's selected set, ranked by `eff_per_dollar` (adjusted_expected / optimizer_cost). Use it when:

- You lose a target movie above MaxInt — immediately scan alternates for the best available substitute
- Your remaining budget is small (< $30) — alternates with high eff_per_dollar at low cost may still be worth bidding
- The `Score > median(pool)` and `P(Edge) > 0.45` rule from the strategy guide applies here: if no alternates pass both thresholds, hold the remaining cash

**eff_per_dollar is the knapsack shadow price**: items with higher eff_per_dollar provide more adjusted expected value per auction dollar spent. The alternates list is pre-sorted so the top row is always the highest-value substitute for your next bid.

**Unspent budget rule**: If you have completed your anchor targets and more than 15% of budget (~$30) remains with no alternates meeting both Score and P(Edge) thresholds, do not force spend. Unspent budget is better than negative-edge exposure. This is consistent with auction theory's finding that budget-preservation in sequential auctions is sometimes the dominant strategy.

---

## 7) Two-Pass Bid Confirmation

For each contested movie where you are unsure whether to keep bidding:

1. **Pass 1 — market_fair_bid**: confirms the movie survives competitive pricing. If it remains in the optimizer's selected set at market_fair_bid cost, it is worth fighting for.
2. **Pass 2 — target_bid**: confirms your personal execution level. Read TgtInt as your comfortable anchor and MaxInt as your absolute ceiling.

If a movie fails Pass 1 (optimizer drops it under market_fair_bid pricing), it means the movie consumes too large a share of the league budget to be worth fighting for at those prices. Pivot to alternates.

If a movie passes Pass 1 but your current bid exceeds TgtInt, you are in the TgtInt–MaxInt zone. Re-read win probability: if it holds up under the aggressive-bidup opponent profile, you can continue bidding. If win probability drops sharply, stop.

---

## 8) Common In-Draft Mistakes

1. **Using expected_gross as the primary objective**: EV maximization ignores opponent positioning. Two portfolios with the same expected gross can have very different win probabilities depending on concentration and overlap with opponents.

2. **Not updating opponent budgets**: The most common stale-data error. If three opponents blow their budgets in Cycle 1, Cycle 2 movies become significantly cheaper — but only if the model knows about it. Always update remaining_money for everyone after each assignment.

3. **Enforcing prices at the cost of your own cycle budget**: Bidding up a movie you don't want costs you real budget. Only enforce if you are above your cycle budget target.

4. **Bidding above MaxInt**: MaxInt = min(market_fair_bid, personal_remaining). Above this, you are paying more than the league collectively values the film. There is no expected-value case for this.

5. **Treating win probability differences under 2pp as meaningful**: With 10,000 MC samples and p≈0.20, the standard error is ±0.4pp. Differences under ~2pp are noise — do not change your portfolio composition based on them.

6. **Misreading TgtInt as a bid target for the live room**: TgtInt is your *personal budget fraction*, not the competitive market price. For Avengers, TgtInt might show $17 (your fraction of your $200 budget), while MaxInt shows $141 (market fair bid capped by personal budget). The live bid will be somewhere between these. Use MaxInt as your real ceiling; use TgtInt only to confirm the movie is within your value anchor before the bid escalates to competitive levels.

7. **Skipping reruns after anchor wins**: After winning a high-cost movie, your remaining budget drops sharply. The optimizer needs to re-run to find the best portfolio under your new constraint. Stale recommendations from a pre-win run may propose movies you can no longer afford.

---

## Sources

- [Sequential auctions with budget constraints — fantasy basketball evidence (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S2214804316300106)
- [Dynamic Auctions with Budget-Constrained Bidders (Kobayashi, 2023)](https://www.shuntokobayashi.com/assets/files/shunto-kobayashi-jmp.pdf)
- [Knapsack-based portfolio optimization under uncertainty (PLOS One)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0213652)
- [T-copula tail dependence in portfolio risk (Ideas/JMVA)](https://ideas.repec.org/a/eee/jmvana/v191y2022ics0047259x22000471.html)
- [Copula-GARCH portfolio optimization (Frontiers)](https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2025.1675120/full)
