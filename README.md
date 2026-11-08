# Castle Challenge

My work on Sequoia's No Limit Castle Challenge, a Colonel Blotto variant. It happened in two
phases. First I simulated the field and picked an allocation blind. Then I found the public live
leaderboard and switched to measuring the field instead of modeling it. The second phase is most of
this repo, and it's where the score improved.

| | Blind pick | Score, rank | Final pick | Score, rank |
|---|---|---|---|---|
| **Game 1** (head to head) | 8-17-30-44-1 | 8.68, #1,140 | **4-7-35-47-7** | **9.288, #30** of 5,915 |
| **Game 2** (4-player groups) | 18-8-2-2-70 | 5.00-5.05, ~#700 | **16-2-2-2-78** | **5.62-5.70, #110-173** of 5,394 |

Scores are from the live board on Nov 8, 2026 (last update 19:51 UTC). Both picks are scored
against the same field. The top scores were 9.312 and about 5.83.

- **Game 1 is stable.** The final split has scored 9.287-9.288 and ranked #29-30 on every board
  since it went in. 18 other entries share its exact score, so it's tied for #30.
- **Game 2 isn't.** The site re-simulates Game 2 with fresh random groups at every update, and
  the field is packed tight around this score. The same entry has ranked anywhere from #110 to
  #173 (see [Why Game 2 was so hard](#why-game-2-was-so-hard)).

## The rules

You allocate 100 soldiers across five castles worth 1-5 points (15 total).

- **Game 1:** you play head to head against every other entry. More soldiers wins the castle and
  ties split it. Your score is your average points per matchup.
- **Game 2:** random groups of four. The most soldiers takes each castle, then players are ranked by
  points and paid 9 / 4 / 2 / 0, with ties splitting payouts.

## The math

**Game 1 separates by castle.** Let $X_c$ be the soldiers a random opponent sends to castle $c$,
and $v_c = c$ its value. Against a field, a split $a$ scores

$$S(a) = \sum_{c=1}^{5} v_c\, G_c(a_c), \qquad G_c(x) = \Pr(X_c < x) + \tfrac{1}{2}\Pr(X_c = x).$$

Only the five per-castle distributions matter, so the exact best response is a knapsack over the
castles:

$$B_k(t) = \max_{0 \le s \le t} \Big[ B_{k-1}(t-s) + v_k\, G_k(s) \Big], \qquad S^{*} = B_5(100).$$

That's cheap, and it's checked against all $\binom{104}{4} = 4{,}598{,}126$ legal splits.

**One-soldier tests are exact.** Moving one soldier from castle $i$ to castle $j$ changes the
score by

$$\Delta_{i \to j} = v_j \Big[ G_j(a_j + 1) - G_j(a_j) \Big] - v_i \Big[ G_i(a_i) - G_i(a_i - 1) \Big].$$

Moves on different castles add: $S(a + \delta_1 + \delta_2) = S(a + \delta_1) + S(a + \delta_2) - S(a)$.
That's how 3-6-36-48-7 gets scored without an entry (below).

**Player models.**

- Thinking levels follow a Poisson distribution, $f(k) = e^{-\tau}\tau^k / k!$. Level $k$
  best-responds to levels $0, \dots, k-1$, mixed in proportion to $f$. This is the cognitive
  hierarchy model.
- Noisy players use a logit response over all 4.6M splits, $p(a) \propto e^{\lambda S(a)}$. It's
  computed exactly by the same knapsack with $(\max, +)$ swapped for $(\mathrm{logsumexp}, +)$.
- The equilibrium comes from fictitious play. Game 1 is constant-sum with value 7.5, so
  $\varepsilon_t = \max_a S(a; \bar\sigma_t) - 7.5$ measures how exploitable the average strategy
  still is.

**Choosing a split without knowing the world.** Worlds $\omega$ have prior weights $\pi_\omega$.
A split's regret in a world is $R_\omega(a) = \max_{a'} S_\omega(a') - S_\omega(a)$. Three ways to
pick:

$$a_{\text{Bayes}} = \arg\max_a \sum_\omega \pi_\omega S_\omega(a), \qquad
a_{\text{minimax regret}} = \arg\min_a \max_\omega R_\omega(a), \qquad
a_{\text{maximin}} = \arg\max_a \min_\omega S_\omega(a).$$

**Fitting a world to anchors (exponential tilting).** Let $u(a, b)$ be the points $a$ takes head
to head from $b$. A Game 1 score is linear in the field's weights, so each anchor $(a_j, s_j)$ is a
linear constraint. Of all fields that match the anchors, take the one closest to the prior:

$$\min_w \ \mathrm{KL}(w \,\|\, \pi) \quad \text{s.t.} \quad \sum_b w_b\, u(a_j, b) = s_j \ \text{ for every } j
\qquad \Longrightarrow \qquad w_b \propto \pi_b \exp\Big( \sum_j \theta_j\, u(a_j, b) \Big).$$

Each $\theta_j$ is solved by safeguarded Newton. The constraint's derivative in $\theta_j$ is
$\mathrm{Var}_w[u(a_j, \cdot)]$. The solver cycles over the anchors until all of them hold.

**Weighting worlds.** Refit each world $m$ without anchor $j$, predict that anchor, and call the
error $e_{mj}$. Then

$$\omega_m \propto \exp\Big( -\frac{1}{2\sigma^2} \sum_j e_{mj}^2 \Big), \qquad \sigma = 0.5.$$

**Reading newcomers from score changes.** A Game 1 score averages over the other $N - 1$ entries.
When $k$ entries join between snapshots, an anchor's total moves by exactly its points against
them:

$$(N_t - 1)\, s_t - (N_0 - 1)\, s_0 = \sum_{b \in \text{new}} u(a, b).$$

Dividing by $k$ gives the anchor's average result against the newcomers. The site rounds scores to
3 decimals, so this read has standard error
$\sqrt{(N_0 - 1)^2 + (N_t - 1)^2} \cdot \frac{0.001}{\sqrt{12}} \big/ k$. That's about 0.1 for
26 newcomers, and it shrinks as $1/k$. The reads are fit with a soft version of the tilt,

$$\min_w \ \mathrm{KL}(w \,\|\, \pi) + \sum_a \frac{\big( \mathbb{E}_w[u_a] - m_a \big)^2}{2\, \mathrm{sd}_a^2},$$

solved in its dual by damped Newton.

**Game 2.**

- Castle points total 15, so more than 7.5 points guarantees first place. Sweeping any set of
  castles worth 8 or more (like {5, 3} or {4, 3, 1}) locks in the 9 payout.
- Beyond that there's no closed form, because payout depends on rank among four players. Splits
  were found by search.
- The phase 1 equilibrium started with replicator dynamics, $x_i \leftarrow x_i\, e^{\eta (u_i - \bar u)}$.
  Then it switched to fictitious play with a search-based best response.
- Live Game 2 scores are averaged over distinct board updates, with standard error
  $\hat\sigma / \sqrt{n}$. A gap between two splits counted only if it beat the update-to-update
  noise of about 0.02.

## Phase 1: simulating the field (blind)

Game 1 depends only on how many soldiers the field sends to each castle, so the best response to any
assumed field is exact (a knapsack over the five castles), and every one of the 4.6M legal
allocations can be checked. I modeled the field as a mix of:

- habit players (round numbers, even splits, "stack the 5")
- level-k thinkers
- equilibrium players (fictitious play, which matches the known continuous Blotto solution)
- crowd-modelers

That gave 180 "worlds", which I weighted with a prior. I wrote two priors: a sharp main one and a
skeptical baseline. The full study is in `sim/experiments.py` and `results/simulation/`.

- **Regret.** I scored every split in every world, then compared the three choice rules from
  [The math](#the-math).
  - The Bayes pick (8-17-30-44-1, 8.69 expected) and the minimax-regret pick (5-19-31-45-0, 8.63)
    both conceded castle 5.
  - The maximin pick (5-5-28-44-18) hedged with 18 on castle 5 and lost 0.74 on average.
  - I went with Bayes.
- **Crossover.** A shallow enough field flips the answer to stacking castle 5. That bet was
  lopsided, though: conceding castle 5 cost 0.08 if wrong, and stacking it cost 0.60.
- **Focal points.** One soldier above crowded round numbers flips many castles. That idea came
  back later as "one above the cluster."
- **Game 2.** I tried replicator dynamics on a fixed pool of plans. Its equilibrium stayed
  exploitable: a best response beat it by 3.7 payout points. Fictitious play with a search-based
  best response cut that gap to 0.38.
- **Counter-players.** I added players one level above the crowd-modelers, at 10% and 25% of the
  field. Game 1 held up. Game 2 slipped at 25%.

The answer was to concede castle 5 and win the other four (8-17-30-44-1).

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
- **Ranks swing more than scores.** Around 5.65 the field is packed: ranks 101-154 span only 0.053
  points. So 16-2-2-2-78 has scored 5.62-5.70 but ranked anywhere from #110 to #173. A batch of
  test entries from one team landing just above it accounted for about 11 of those places. The
  blind pick swings the same way, between #681 and #726.
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

## After the deadline: can you read other players' splits off the board?

An idea I didn't get to during the challenge: use the anchors and fitted worlds to infer what
*other* entries played, with a confidence score, from nothing but their Game 1 score. Knowing the
#1 split, for instance, would have been worth testing against. I built it afterwards in
`live/infer_splits.py`.

### The method

Every fitted world $m$ predicts the score $S_m(a)$ of all 4.6M splits and says how common each
split is, $w_m(a)$. For an entry with board score $s$:

$$P(a \mid s) \;\propto\; \sum_m \pi_m \Big[ (1 - \epsilon)\, w_m(a) + \frac{\epsilon}{4.6\text{M}} \Big] \, \mathcal{N}\big(s;\ S_m(a),\ \sigma^2\big).$$

- $\pi_m$ is the world's holdout weight.
- The floor $\epsilon$ lets a split the library never sampled win if its score fits better.
- $\sigma$ is the model's score error.
- Splits with a measured score (anchors) get an exact likelihood instead: certain at their own
  score, impossible anywhere else.

Sorting each world's 4.6M predicted scores once makes every query a single slice. That made it a
few seconds of compute, not the long grid search I expected.

$\epsilon$ and $\sigma$ are tuned per rank band on 8 synthetic boards with a known field. Each
board has anchors shaped like mine: five corners, a posted split, and 10 test entries climbing
from it. The settings are tuned by log score of the true split, a proper scoring rule, so
confident misses are punished. Then they're tested on the 23 real anchors, each held out in
turn. The real anchors are never used for tuning.

### Is the information there? Yes, at the top

With the *true* field known, I counted how many legal splits share each entry's rounded score on
synthetic boards:

| Ranks | Legal splits sharing the score (median) |
|---|---|
| 1-10 | **1** |
| 11-50 | 2-3 |
| 51-200 | 9-10 |
| 201-1000 | 50-72 |
| 1001-3000 | 790-940 |
| 3001+ | about 1,280 |

So a perfect model could name the exact split of each top-10 entry from its score alone. That
needs predicted scores good to about 0.0005, though, the site's rounding.

### Can this model read it? No

With each real anchor held out, the fitted worlds missed its score by 0.006-0.06 points near the
top (ranks 32-394) and 0.06-0.63 below. Even where my test entries were densest, that's about 60×
too coarse. The posterior spreads over about 150 plausible splits at the top and thousands
mid-board.

| Synthetic, by rank | Exact split right | Best guess within 1 soldier | Stated confidence | 90% castle ranges cover truth | Range width | Concede-5 vs stack-5 right |
|---|---|---|---|---|---|---|
| 1-10 | 12% | 57% | 7% | 94% | 13 | 88% |
| 11-50 | 2% | 32% | 7% | 91% | 12 | 88% |
| 51-200 | 0% | 2% | 3% | 97% | 19 | 100% |
| 201-1000 | 0% | 0% | 3% | 94% | 28 | 93% |
| 1001-3000 | 7% | 8% | 7% | 97% | 33 | 92% |
| 3001+ | 13% | 13% | 18% | 91% | 36 | 77% |

- **Real anchors held out:** 1 exact hit in 23, and that was `100-0-0-0-0`, the only split that
  scores that low. 2 of the 23 had a best guess within one soldier.
- **Castle ranges:** the 90% ranges covered 91% of real castles, but they averaged 27 soldiers
  wide near the top.
- **Concede-castle-5 call:** it pointed the right way for 9 of 10 near-top anchors, but mostly at
  52-56%, barely better than a coin flip.
- **Calibration:** stated confidence is honest, slightly *below* the actual hit rate. It's just
  low.

### What it says about the top of the board

| Board | Best guess | P(exact) | P(within 1 soldier) | 90% range per castle | P(concede castle 5) |
|---|---|---|---|---|---|
| #1, 9.311 (7 tied) | 3-11-33-46-7 | 4% | 15% | 2-7 / 6-11 / 32-36 / 2-48 / 6-57 | 86% |
| #32, 9.286 (19 tied) | 4-7-35-47-7 | 70% | 74% | known split | 94% |
| #124, 9.227 (23 tied) | 5-7-35-47-6 | 83% | 83% | known split | 92% |

The #1 split probably has 2-7, 6-11 and 32-36 on castles 1-3. Castles 4 and 5 aren't resolved:
they're either about 46 and 7, or about 2 and 56. The measured clusters come back right only
because their splits are anchors.

Before the exact-anchor rule, the model put the posted 5-7-35-47-6 at #1. That split is known to
score 9.227, 0.084 below #1. A wide error bar plus a big prior weight is enough to claim a split
you've already measured somewhere else. Measured splits now only count at their measured score.

**Takeaway:** the board holds enough information to name the top splits. Reading it would need a
field model about 60× more precise than one built from 23 anchors. One input I didn't use is
cluster size. A 35-way exact tie mid-board is almost certainly a copied chatbot or round-number
split, but the model ignores how many entries share a score. Modeling that would help the big
mid-board clusters, not the top.

## Everything I tried

**Kept** means it shaped the final picks, **Dropped** means I abandoned it, and **Check** means it
ran as a sanity check without deciding anything. More detail on each is in
[`docs/everything_we_tried.md`](docs/everything_we_tried.md).

| Method | Verdict | What happened |
|---|---|---|
| **Phase 1: theory and simulation** | | |
| Exhaustive search + exact knapsack best response | Kept | Used everywhere |
| Fictitious play for the Game 1 equilibrium | Kept | Matches the known continuous Blotto solution |
| Cognitive hierarchy + logit noise + habit players | Kept | Became the prior library for phase 2 |
| 180 worlds under two written priors | Kept | Every weight came with a reason |
| Bayes vs minimax-regret vs maximin selection | Kept (Bayes) | Maximin hedged castle 5 and lost 0.74 on average |
| Crossover analysis on field depth | Kept | Conceding castle 5 was the safer bet (0.08 vs 0.60 if wrong) |
| Focal-point steps | Kept | Became "one above the cluster" |
| Replicator dynamics for Game 2 | Dropped | Exploitable by 3.7; fictitious play cut that to 0.38 |
| Counter-player stress test | Check | Game 1 held; Game 2 slipped at 25% counter-players |
| **Phase 2: inferring the field** | | |
| Fitting the score histogram (EM over player types) | Dropped | Lost 1-2.5 points on synthetic boards |
| Anchors + exponential tilting | Kept | About 40 anchors cut the loss to 0.02 |
| Board-top constraint | Dropped as a selector | Predicted 9.316 for a split that measured 9.105 |
| 12-world ensemble weighted by holdout error | Kept | Ranks candidates |
| Soft anchors | Dropped | Redundant once hard anchors existed |
| 300 hidden late-entrant scenarios | Kept | 100-600 entrants, 20-60% board-trackers |
| Objective: P(#1) → mean score | Kept (switched) | Validation showed P(#1) was overconfident |
| Histogram world matching (Wasserstein) | Check | Didn't change the pick |
| Synthetic validation | Check | Ensemble median rank 161 vs oracle 34, so final picks came from measurements |
| **Watching the crowd** | | |
| Chatbot answers as their own population | Kept | No effect on the final choice |
| Fingerprinting chatbot splits on the board | Check | Not significant (p ≈ 0.13) |
| Copy-cluster tracking | Kept | Showed copying wasn't happening |
| Newcomer tracking from anchor score changes | Kept | About 12% of newcomers tracked the board |
| Level-k escalation off the posted split | Check | Cycles between conceding and stacking castle 5, not a ladder |
| Searching online for the #1 split | Dropped | Never found it |
| Cross-game fingerprinting | Check | Top Game 2 entries with Game 1 scores of 2.6-4.8 used a throwaway |
| Leak check | Check | Largest exact tie in the Game 2 top 500 was 6 entries |
| Decay check | Check | Game 1 drifted about −0.002 per hour |
| **Test entries** | | |
| A portfolio of 8 strong splits | Dropped | Replaced by one-change tests |
| Value-of-information test design | Dropped | Favored splits that told worlds apart, not ones that could win |
| One-soldier tests + separability | Kept | Exact measurements with no model |
| Adaptive design (each test one step from the best) | Kept | Found 4-7-35-47-7 |
| **Game 2 modeling** | | |
| Averaging over board updates with standard errors | Kept | Fixed a bug where republished boards shrank the error bars |
| Five-population mixture fit | Dropped | Top claim measured −0.15; best tweak had the wrong sign |
| Common random numbers in simulation | Kept | Lower-variance comparisons between candidates |
| Decide on measured scores only, gap > 0.02 | Kept | Picked 16-2-2-2-78 |
| **After the deadline** | | |
| Identifiability ceiling (true field, splits per rounded score) | Check | Top-10 scores each belong to one legal split |
| Posterior over other entries' splits from their scores | Negative result | Honest confidence, but exact splits unreadable: model error 0.03 vs 0.0005 needed |
| Exact likelihood for measured splits | Kept (fix) | Stopped the model claiming a measured split at the wrong score |
| Cluster size as evidence of a copied split | Untried | Next lever, for mid-board copy clusters only |

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
  infer_splits.py       posterior over other entries' splits (after the deadline)
tests/                  87 tests
docs/                   everything_we_tried.md: full log of every method, kept or dropped
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
.venv/bin/python -m live.infer_splits --validate --infer # ~30 min; needs data/anchors.csv
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
