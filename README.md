# Castle Challenge

My work on Sequoia's No Limit Castle Challenge, a Colonel Blotto variant. It happened in two
phases. First I simulated the field and picked an allocation blind. Then I found the public live
leaderboard and switched to measuring the field instead of modeling it. The second phase is most of
this repo, and it's where the score improved.

| | Blind pick | Score, rank | Final pick | Score, rank |
|---|---|---|---|---|
| **Game 1** (head to head) | 8-17-30-44-1 | 8.68, #1,132 | **4-7-35-47-7** | **9.287, #30** of 5,904 |
| **Game 2** (4-player groups) | 18-8-2-2-70 | 5.03, ~#700 | **16-2-2-2-78** | **5.68, ~#110-155** of 5,375 |

Scores are from the live board on Nov 8, 2026. Both picks are scored against the same field. The
top scores were 9.310 and about 5.85. Game 2 scores are averaged over several board updates
because the site re-simulates them each time (see below).

## The rules

You allocate 100 soldiers across five castles worth 1-5 points (15 total).

- **Game 1:** you play head to head against every other entry. More soldiers wins the castle and
  ties split it. Your score is your average points per matchup.
- **Game 2:** random groups of four. The most soldiers takes each castle, then players are ranked by
  points and paid 9 / 4 / 2 / 0, with ties splitting payouts.

## Phase 1: simulating the field (blind)

Game 1 depends only on how many soldiers the field sends to each castle, so the best response to any
assumed field is exact (a knapsack over the five castles), and every one of the 4.6M legal
allocations can be checked. I modeled the field as a mix of:

- habit players (round numbers, even splits, "stack the 5")
- level-k thinkers
- equilibrium players (fictitious play, which matches the known continuous Blotto solution)
- crowd-modelers

That gave 180 "worlds", which I weighted with a prior. The answer was to concede castle 5 and win
the other four (8-17-30-44-1). The full study is in `sim/experiments.py` and `results/simulation/`.

It got the *shape* of the answer right, but it missed by about 0.6 points. The real field doesn't
spread out smoothly. It bunches on a few exact numbers, and you can only see that in real data.

## Phase 2: measuring the field

### The leaderboard is data

The leaderboard page loads one JSON file with every entry's name, school, score and rank for both
games. It doesn't include allocations. Scores are recomputed against the whole pool every time the
file regenerates (every 15-45 minutes). `live/scrape_leaderboard.py` polls it every 5 minutes and
saves each new version.

Three facts make it useful:

1. **An exact tie means an identical split.** A Game 1 score is an exact function of the split, so a
   cluster of entries with the same score to three decimals is a group of copies. One team posted
   5-7-35-47-6. Its cluster held 22-24 entries the whole time, which confirmed the split.
2. **Known splits are anchors.** Any entry whose split I know gives me (split, live score). That
   pins down the field far better than the score histogram does.
3. **Score changes reveal newcomers.** If an anchor's score moves between snapshots, the change is
   its total payoff against the people who joined in between. That tells you how newcomers play,
   without seeing their splits.

### Why the score histogram alone isn't enough

My first instinct was to fit a model of the field to the shape of the score histogram. On synthetic
leaderboards where I knew the true field, that fit picked allocations that lost 1-2.5 points. Many
different fields produce nearly the same histogram. Anchors don't have that problem. With about 40
anchors near the top, the loss on synthetic boards dropped to 0.02 points.

Anchors came from three places:

- splits people published in GitHub repos and posts, matched to their board entries
- entries that put all 100 soldiers on one castle, found online
- test entries designed to measure one change at a time (below)

All of them, with live scores but no names, are in `results/live/measured_splits.csv`.

### Worlds, and picking between them

`live/robust.py` builds 12 candidate fields: 3 thinking depths × 4 population mixes (balanced,
sharp-heavy, equilibrium-heavy, habit-heavy).

- **Fit to the anchors.** Each world is reweighted by exponential tilting, the smallest change from
  its prior that reproduces every anchor's live score exactly.
- **Weight by holdout accuracy.** Each world's weight comes from leave-one-out error: refit without
  one anchor, then check how well the world predicts it.
- **Add hidden late entrants.** Across the worlds I sample 300 scenarios of late entrants I can't
  see: 100-600 of them, 20-60% of whom track the board like I was.
- **Rank by mean score.** Candidates are ranked by their average score across all scenarios.

Two ways of picking worlds that I tested:

- **Match the full score histogram.** For each world I compared its simulated score distribution
  with the real board's (Wasserstein distance).
  - Habit-heavy worlds fit worst (0.16-0.18 off) and sharp-heavy worlds fit best (0.10).
  - Weighting by histogram fit didn't change the pick. It's a useful sanity check, not a decider.
- **Pick the world where your best answer scores highest.** This is a trap.
  - Forcing every world's best response to reach the board's top score made the model predict
    9.316 for a stack-castle-5 split. Its measured score was 9.105.
  - One measurement flipped the ensemble's weight on "concede castle 5" worlds from 0 to 1.
  - After that the model was accurate. It predicted 9.288 for 4-7-35-47-7, which measured 9.288.

### Accounting for LLM randomness

Lots of applicants paste the prompt into a chatbot, and chatbots don't pick numbers at random. I
collected 156 real chatbot answers to the prompt (credits below). They were very repetitive:

- Only 98 were distinct.
- 7-13-20-27-33 alone was 18% of them (28 of 156).
- The rest clustered on "nice" arithmetic like 10-15-20-25-30 or 5-10-20-30-35.

I added them to every world as their own population (10% of the field) instead of treating them as
noise. I also tried to find them on the board: their predicted scores do land on exact-tie spikes
(7-13-20-27-33 is predicted at 6.660, and there's a spike at 6.663). But there are 49 spikes between
6.5 and 7.2, so a match is weak evidence: 10 of the top 20 answers matched within 0.003, against
about 7 expected by chance (p ≈ 0.13).

It didn't matter for the final choice. Chatbot splits put 30+ soldiers on castle 5, so every
concede-castle-5 finalist wins and loses exactly the same castles against them.

### Test entries and one-soldier changes

This was the most valuable step. Game 1 scoring separates by castle:

```
score(split) = sum over castles c of  value_c × G_c(soldiers on c)
G_c(x) = share of the field you beat on castle c with x soldiers (ties count half)
```

So two splits that differ by one soldier differ in score by exactly the value of that one move
against the real field. Moves on different castles add exactly. A test entry that differs from a
measured split by one soldier therefore measures that move exactly (to ±0.001, from rounding), with
no model involved.

I designed the test entries this way. Each one was also a real candidate for the final pick, one
step from the best measured split. Measured effects:

| Change | From → to | Effect |
|---|---|---|
| castle 1 → castle 5 (5th castle 6 → 7) | 5-7-35-47-6 → 4-7-35-47-7 | **+0.057** |
| castle 1 → castle 3 (35 → 36) | 4-7-35-47-7 → 3-7-36-47-7 | −0.010 |
| castle 4 → castle 5 (48/7 → 47/8) | 2-8-35-48-7 → 2-8-35-47-8 | −0.026 |
| castle 2 → castle 4 (47 → 48) | 4-7-35-47-7 → 4-6-35-48-7 | −0.046 |
| castle 5 → castle 4 (48 → 49) | 2-8-35-48-7 → 2-8-35-49-6 | −0.071 |
| castle 5 → castle 4 (49 → 50) | 2-8-35-49-6 → 2-8-35-50-5 | −0.070 |
| concede castle 5 → stack castle 5 | 2-8-35-48-7 → 4-8-27-3-58 | −0.099 |
| castle 1 → castle 5 (8 → 9) | 2-8-35-47-8 → 1-8-35-47-9 | −0.142 |

Separability also produced results without any new entry:

- **A combination for free.** 3-6-36-48-7 combines two measured moves on four different castles, so
  its score is 9.277 + 9.241 − 9.287 = **9.231**, worse than either.
- **Ruled out a candidate.** 3-7-35-47-8 differs from 3-7-35-48-7 by the same castle-4/castle-5
  swap as the 2-8-35 pair, so it's exactly 0.026 worse.
- **A lower bound that predicted the winner.** The 48 → 49 result shows that going from 6 to 7 on
  castle 5 is worth at least +0.07 (the castle 4 part of that move can only cost). That's why
  4-7-35-47-7 was the right next test.

I also tried choosing tests by value of information across the 12 worlds. It favored splits that
told the worlds apart, not splits that could win. Since the worlds had all agreed (wrongly) an hour
earlier, I dropped it in favor of the rule above: every test should be a step from the best measured
split.

### Other people tracking the board

I assumed I wasn't the only one waiting and tracking the board, so the late-entrant scenarios
include sharp players who best-respond to it at one, two and three levels deep. I also tracked what
actually happened:

- **Newcomers.** From anchor score changes, roughly 12% of new entries behaved like board trackers.
  In one 6-hour window, 2 of the 29 newcomers who weren't my test entries landed exactly on the
  #1 score.
- **Copying.** I checked the anti-copier argument ("late entrants will copy the visible top split,
  so sit one soldier above it") against the copy clusters. Over 9 hours, the posted split's cluster
  stayed at 22-24 and 4-7-35-47-7's at 15-16. Copying wasn't happening, so I took the higher
  measured score (4-7-35-47-7, 9.287) over its anti-copier variant (3-7-36-47-7, 9.277).
- **Testing teams.** Some groups ran 10-20 test entries each. One team's 15 Game 1 entries all sit
  in the top 150, while they threw away their Game 2 entries. Another group holds several spots in the
  Game 2 top 30.

### Validation

`live/validate.py` builds synthetic leaderboards with a known field and hidden late entrants, and
gives each method only what I'd have had: anchors, posted splits and scores. Median final rank over
12 trials:

| Method | Median rank |
|---|---|
| Best response to the true existing field (oracle) | 34 |
| Ensemble of worlds + late entrants (this repo) | 161 |
| Best response to a single fitted field | 241 |
| Best response matched to the board top | 275 |
| Posted leader with its best one-soldier change | 324 |
| Copy the posted leader | 368 |

The ensemble beat the simple approaches but was nowhere near the oracle. That gap is why the final
picks came from measured test entries, not from the model.

## Why Game 2 was so hard

Game 1 improved by about 0.6 points through careful measurement. Game 2 fought back at every step:

- **It doesn't separate.** A castle goes to the highest of four players, and payouts depend on your
  *rank* by total points. The value of a soldier depends on the other three players jointly, so
  there's no per-castle G-function and no exact best response, only search.
- **The board is noisy.** Game 2 scores are re-simulated with fresh random groups at every update,
  so the same entry moves about ±0.02 between updates. Each test needed 3+ updates to read
  reliably, compared with one exact read in Game 1.
- **It has cliffs.** 20-2-2-2-74 averaged 5.45 and 20-0-0-0-80 averaged **2.92**. Taking the 2s off
  castles 2-4 cost 2.5 points. Many castle-5 stackers leave the middle castles at 0-1, so 2 soldiers
  win them. Without the 2s you can't get much past 6 points, and anyone who puts 1s on the middle
  castles beats you.
- **Models were wrong even about direction.** A mixture model fit to the Game 2 anchors missed some
  of them by up to 0.95.
  - Its top claim (13-38-49-0-0) measured 0.15 *worse* than the split it was meant to beat.
  - Its best local tweak (20-2-3-3-72, predicted +0.13) measured −0.10.
- **Game 1 scores don't reveal Game 2 splits.** Many of the top Game 2 entries have Game 1 scores of
  2.6-4.8, which no sensible Game 2 split produces in Game 1. They put a throwaway in Game 1.
- **There's no public answer to anchor on.** No large exact-tie cluster exists near the Game 2 top,
  and the top is mostly groups running many test entries.

What did work, all measured (averages over board updates):

| Change | From → to | Effect |
|---|---|---|
| castle 1 → castle 5 (70 → 74) | 18-8-2-2-70 → 14-8-2-2-74 | **+0.32** |
| castle 2 → castle 1 | 14-8-2-2-74 → 20-2-2-2-74 | +0.10 |
| castle 1 → castle 5 (74 → 78) | 20-2-2-2-74 → 16-2-2-2-78 | **+0.22** |
| castle 5 → castles 3 and 4 | 20-2-2-2-74 → 20-2-3-3-72 | −0.10 |
| three castles instead of castle 5 | 14-8-2-2-74 → 13-38-49-0-0 | −0.15 |
| drop the 2s for castle 5 | 20-2-2-2-74 → 20-0-0-0-80 | −2.53 |

Game 2 is a race for castle 5 among the people who stack it, with cheap 2s to sweep the middle
castles they leave empty. The race was still paying at 78. 82 was the next step, but I didn't submit
anything untested in a game with cliffs like that.

## What I'd take away

- Measure the field instead of modeling it, whenever the scoring lets you. Separable scoring turns
  each one-soldier test into an exact measurement.
- Use the model to choose the next test, not to make the final pick.
- Agreement inside an ensemble isn't evidence. All 12 worlds agreed and were wrong.
- Check strategic assumptions against the data before paying for them. The anti-copier argument
  sounded right, but the copy clusters weren't growing.

## Repo layout

```
sim/                    offline study
  castles.py            rules, exact Game 1 best response, Game 2 payouts and search, player models
  experiments.py        the 180-world simulation study (phase 1)
live/                   leaderboard pipeline
  scrape_leaderboard.py snapshots of the public leaderboard JSON
  anchors.py            known splits matched to board entries
  board_fit.py          prior library of player types, exponential tilting to anchors
  robust.py             ensemble of worlds, late-entrant scenarios, candidate ranking
  newcomers.py          newcomer behavior from anchor score changes
  probes.py             model-free comparisons between test entries
  strategy.py           runs the pipeline on the newest snapshot
  game2.py              Game 2 anchor averaging and mixture fit
  validate.py           synthetic-leaderboard validation
tests/                  80 tests
results/simulation/     phase 1 outputs (baseline/ holds a more skeptical prior)
results/live/           measured splits and validation results
data/                   gitignored: raw snapshots (thousands of students' names), anchor ids
```

## Run it

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                            # ~10 s
.venv/bin/python -m sim.experiments                      # phase 1 -> results/simulation/
.venv/bin/python -m live.scrape_leaderboard --once       # one snapshot -> data/snapshots/
.venv/bin/python -m live.strategy --once                 # needs data/anchors.csv
.venv/bin/python -m live.validate --trials 12            # -> results/live/validation.csv
```

`data/anchors.csv` has columns `board_id,game,label,alloc,source` and isn't committed, because it
ties splits to named entries. Raw snapshots stay local for the same reason. Only aggregates are
published.

## Credits

Public splits and data I used as anchors or populations:

- [blobberus/CASTLE](https://github.com/blobberus/CASTLE): chatbot answers to the prompt
- [AdenNowell/Allocation-Study-Sequoia](https://github.com/AdenNowell/Allocation-Study-Sequoia)
- [HaykGar/CastleChallengeCitadel](https://github.com/HaykGar/CastleChallengeCitadel)
- [lachlanspangler/CastleChallenge](https://github.com/lachlanspangler/CastleChallenge)
- [nischalon10/poker-bot-tournament](https://github.com/nischalon10/poker-bot-tournament)
- [qbeka/No-Limit-a-bot-building-tournament-simulations](https://github.com/qbeka/No-Limit-a-bot-building-tournament-simulations)
- [Mianmalol/No_Limit_Bot_Tournament](https://github.com/Mianmalol/No_Limit_Bot_Tournament)
