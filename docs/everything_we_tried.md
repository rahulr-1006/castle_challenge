# Everything I tried

A complete log of the methods behind this repo, roughly in the order I tried them, with what each
one did. **Kept** means it shaped the final picks, **Dropped** means I abandoned it (and why), and
**Check** means it ran as a sanity check without deciding anything. Numbers are from the live board
unless noted.

## 1. Theory and simulation (phase 1, blind)

- **Exact rules engine.** Game 2 castle points are stored in twelfths so 2-, 3- and 4-way splits
  stay exact, which matters because ranks compare totals for equality. **Kept**
- **Exact Game 1 best response.** Game 1 depends only on the field's per-castle distribution, so
  the best response to any field is a knapsack over five castles. **Kept** (used everywhere)
- **Exhaustive search.** Scored all 4,598,126 legal allocations in every world. **Kept**
- **Equilibrium by fictitious play.** Game 1's equilibrium matches the known continuous Blotto
  solution. A separate fictitious play handles Game 2. **Kept** (as one player type)
- **Player models:**
  - level-0 habit players (round numbers, "stack the 5", uniform)
  - a Poisson cognitive hierarchy (level-k with mean depth τ)
  - logit noisy responses, with noise calibrated so strategic players lose about 3% or 11-12% of
    the best score
  - equilibrium players
  - crowd-modelers

  **Kept** (they became the prior library for phase 2)
- **180 worlds with a written prior.** Axes were depth, equilibrium share, crowd-modeler share,
  level-0 style and noise. Every weight came with a reason. I built two priors: "sharp" (main)
  and "baseline" (skeptical). **Kept**
- **Regret-based selection.** For every allocation I computed its prior-expected score, its
  worst-world score (maximin), its prior-weighted regret and its **minimax regret**:
  - the Bayes pick (8-17-30-44-1, 8.69 expected) and the minimax-regret pick (5-19-31-45-0, 8.63)
    both conceded castle 5
  - the maximin pick (5-5-28-44-18) hedged with 18 on castle 5 and lost 0.74 on average

  I went with the Bayes pick. **Kept**
- **Crossover analysis.** I tracked how the best pick changes with field depth. A shallow field
  flips the answer to stacking castle 5, but the bet was lopsided: losing 0.08 if wrong one way,
  0.60 the other. **Kept**
- **Focal-point steps.** One soldier above crowded round numbers flips many castles. **Kept**
  (it became "one above the cluster" later)
- **Game 2 search.** Plan generators, local search, simulated annealing, a 4-way castle-share
  model and "concentrate" variants. **Kept** (search only, since there's no exact solution)
- **Replicator dynamics for the Game 2 equilibrium.** On a fixed pool of plans it stayed
  exploitable: a best response beat it by 3.7 payout points. Fictitious play with a search-based
  best response cut that gap to 0.38. **Dropped** for fictitious play.
- **Counter-player stress test.** I added players one level above the crowd-modelers (10% and 25%
  of the field). Game 1 held up. Game 2 slipped at 25%. **Check**
- **Blind result:**

  | Game | Pick | Score | Rank |
  |---|---|---|---|
  | Game 1 | 8-17-30-44-1 | 8.70 | #1,093 |
  | Game 2 | 18-8-2-2-70 | 5.03 | #664 |

  The model got the shape of the field right but missed its exact-number spikes.

## 2. Scraping and infrastructure

- **Found the data source.** The leaderboard page loads one public JSON file
  (`/castle_challenge/data/leaderboard.json`) with names, schools, scores and ranks for both
  games, but no allocations. **Kept**
- **Scraper.** It retries, validates the schema, dedupes by the file's `generated_at`, writes
  gzipped snapshots atomically, and keeps an index with new-entry counts. It polled every 15
  minutes, later every 5, from Nov 7 through the deadline, under `caffeinate` so the Mac wouldn't
  sleep. **Kept**
- **Strategy loop.** It reruns the whole pipeline on every new board and writes a run log plus a
  readable report (pick, worlds, anchors, newcomers, test comparisons). **Kept**
- **How the board behaves:**
  - It's one global pool.
  - Every score is recomputed against everyone each time the file regenerates (every 15-45
    minutes).
  - Game 2 is re-simulated with fresh random groups at each update.
  - Identical splits tie exactly.

  **Kept** (every later method depends on these)
- **Privacy.** Raw snapshots (thousands of student names) stay local and gitignored. Only
  aggregates get published. **Kept**

## 3. Inferring the field from scores

- **Fitting the score histogram** (EM over a library of player types). On synthetic boards with
  a known field, it picked allocations that lost 1-2.5 points, because many fields produce the
  same histogram. **Dropped** (kept in the code as a negative result)
- **Anchors**, meaning known (split, live score) pairs, were the breakthrough. On synthetic
  boards, about 40 near-top anchors cut the loss to 0.02 points. Sources:
  - splits published in GitHub repos and posts, matched to board names
  - all-in-one-castle entries found online
  - a posted split confirmed by its 22-24-entry exact-tie cluster and by a second person's
    results card
  - my original entry
  - test entries (section 6)

  **Kept**
- **Name matching** for anchors: full name first, then first name plus last initial among new
  entries. **Kept**
- **Exponential tilting** (the minimum-KL reweighting) of each world so it reproduces every anchor
  exactly, solved with Newton's method. An early version ran too few iterations and produced a
  false "the anchors rule out this family" result. I caught it and fixed it. **Kept**
- **Board-top constraint:** force each world's best response to reach the board's top score.
  It's the "pick the world where your best answer scores highest" idea. It predicted 9.316 for a
  stack-castle-5 split that measured 9.105. **Dropped** as a selector, **Check** only.
- **Ensemble of 12 worlds** (3 depths × 4 population mixes), weighted by leave-one-out anchor
  error. Soft anchors were **Dropped** as redundant once hard anchors existed. **Kept**
- **Hidden late-entrant scenarios:** 300 draws of 100-600 entrants I can't see, 20-60% of them
  sharp board-trackers at levels 1-3, plus copiers of public splits. **Kept**
- **Objective.** I started with P(#1), counting ties. Validation showed P(#1) was overconfident,
  so I **switched to mean score** (with P(top 10) reported). **Kept**
- **Synthetic validation** over 12 trials. Median final rank:

  | Method | Median rank |
  |---|---|
  | Oracle (best response to the true existing field) | 34 |
  | Ensemble | 161 |
  | Single fit | 241 |
  | Board-top tracker | 275 |
  | Posted leader + 1 soldier | 324 |
  | Copy the posted leader | 368 |

  **Check** (it's why the final picks came from measurements, not the model)
- **Histogram world matching.** I compared each world's simulated score distribution with the real
  board's (Wasserstein distance). Habit-heavy worlds fit worst (0.16-0.18) and sharp-heavy best
  (0.10). The pick didn't change. **Check**

## 4. LLM randomness

- **Chatbot answers as a population.** I collected 156 real chatbot answers to the prompt:
  - only 98 were distinct
  - 7-13-20-27-33 alone was 18%
  - the rest sat on "nice" arithmetic like 10-15-20-25-30

  I modeled them as their own 10% population instead of as noise. **Kept**
- **Fingerprinting them on the board.** I matched predicted chatbot scores to exact-tie spikes.
  10 of the top 20 answers landed within 0.003 of a spike, against about 7 by chance (p ≈ 0.13),
  so it wasn't significant. **Check**
- **Effect on the final choice: none.** Chatbot splits put 30+ on castle 5, so every
  concede-castle-5 finalist scores the same against them. **Check**

## 5. Watching the crowd

- **Copy clusters.** Exact ties mean identical splits, so I tracked cluster sizes across every
  update. Over 9 hours:
  - the posted split's cluster stayed at 22-24
  - the cluster at 9.288 (later confirmed as 4-7-35-47-7) stayed at 15-16
  - the #1 score's cluster grew from 3 to 5

  **Kept** (it settled the final Game 1 choice)
- **Newcomer tracking.** When an anchor's score changes between snapshots, the change equals its
  total payoff against the newcomers. Fitting those changes says how newcomers play, about 12% of
  them like board-trackers. Known test entries are subtracted first. **Kept**
- **Level-k escalation off the posted split.** "Beat the leader" reasoning cycles between
  concede-castle-5 and stack-castle-5 (about 57 on castle 5). It doesn't climb steadily, so
  piling more onto castle 5 to stay ahead isn't a ladder. **Check**
- **Searched online for the #1 split.** I didn't find it. **Dropped**
- **Teams running many test entries.** One team's 15 Game 1 entries all sit in the top 150, and
  they threw away Game 2. Another group concentrated on Game 2, and a third spread 10 entries
  widely. **Check**
- **Cross-game fingerprinting.** Every sensible Game 2 split scores 6+ in Game 1, so top Game 2
  entries with Game 1 scores of 2.6-4.8 had entered a throwaway in Game 1. **Check**
- **Leak check.** The largest exact tie anywhere in the Game 2 top 500 was 6 entries, so no
  widely copied answer. **Check**
- **Decay check.** Game 1 drifted about −0.002 per hour. Game 2 rank swings were re-simulation
  noise, because ranks 101-154 span only 0.05 points. **Check**

## 6. Test entries

- **First plan: a portfolio of 8 strong splits.** **Dropped** in favor of designed one-change
  tests.
- **Value-of-information design (expected-regret minimization).** I greedily chose the test that
  most reduced the expected regret of the final pick across 36 ensemble fields. It favored splits
  that told the worlds apart, not splits that could win. Since the worlds had all agreed (wrongly),
  I **dropped** it.
- **Model-free comparisons.** Two entries one soldier apart differ by exactly that move's value
  (±0.001 rounding). **Kept**
- **Separability tricks.** Moves on different castles add exactly, which gave:
  - free scores for untested combinations (3-6-36-48-7 = 9.231)
  - eliminated candidates (3-7-35-47-8 is exactly 0.026 worse than 3-7-35-48-7)
  - lower bounds (castle 5 going 6 → 7 is worth at least +0.07)

  **Kept**
- **Adaptive design.** After each result I re-aimed the next test, and every test was one step
  from the best measured split *and* a real candidate. **Kept**
- **Game 1 path:**
  - first, broad family tests: 33-7-1-10-49 (7.05) and 1-47-19-3-30 (6.01)
  - then the concede-castle-5 core: 2-8-35-48-7 (9.204), 2-8-35-49-6 (9.133), 2-8-35-50-5
    (9.063), 2-8-35-47-8 (9.177), 1-8-35-47-9 (9.036)
  - the stack-castle-5 family test: 4-8-27-3-58 (9.105)
  - next to the posted 5-7-35-47-6 (9.230): **4-7-35-47-7 (9.287)**, 3-7-36-47-7 (9.277) and
    4-6-35-48-7 (9.241)
- **Game 2 path** (averages over updates):
  - the improving path: 18-8-2-2-70 (5.03) → 14-8-2-2-74 (5.35) → 20-2-2-2-74 (5.45) →
    **16-2-2-2-78 (≈5.66)**
  - failed tweaks: 13-38-49-0-0 (5.20), 20-2-3-3-72 (5.36), 20-0-0-0-80 (2.92)
  - other families: 0-37-0-60-3 (4.63), 0-32-0-0-68 (3.98), 0-0-41-0-59 (3.13), 19-37-0-44-0
    (2.67), 10-9-12-37-32 (2.55)

## 7. Game 2 modeling

- **Averaging over updates** with standard errors. Bug found and fixed: when nobody joined, the
  site republished the same Game 2 scores, and counting repeats shrank the error bars. **Kept**
- **Five-population mixture fit** (Game 1 pasters, optimizers, copiers of public splits,
  equilibrium players, chatbot pasters), on a simplex grid with an anchor and board-top loss. It
  missed anchors by up to 0.95, its top claim measured −0.15, and its best local tweak was wrong
  in sign (+0.13 predicted, −0.10 measured). **Dropped** for decisions.
- **Common random numbers** when comparing candidate splits in simulation. **Kept**
- **Final rule: decide on measured scores only, and trust a gap only if it beats the update-to-update
  noise (about 0.02).** **Kept**

## 8. Decisions

- **Final picks:** **4-7-35-47-7** (Game 1, #30 of 5,907) and **16-2-2-2-78** (Game 2, about #155
  of 5,378).
- **Anti-copier rule reversed.** I'd planned to take the variant one soldier above the visible
  clusters if it measured within 0.01 of the best. The cluster data showed copying wasn't
  happening, so I took the higher measured score.
- **Untested ideas I didn't submit:** 82 on castle 5 in Game 2 (the trend was still paying, but
  Game 2 has cliffs), and the unknown #1 Game 1 split, 0.022 above mine.

## 9. Engineering

- 80 tests covering rules, ties, the scraper, anchors, tilting, the ensemble, newcomer tracking and
  test-entry comparisons.
- **Bugs caught along the way:**
  - the tilt ran too few iterations
  - known test entries were counted as unknown newcomers
  - name mismatches on the board
  - a stale module stayed loaded in the long-running loop
  - republished Game 2 boards were double-counted
- **Repo:**
  - `sim/` for phase 1 and `live/` for phase 2
  - results split by phase
  - raw data kept out of git
