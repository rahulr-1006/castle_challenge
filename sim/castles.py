"""Castle allocation game: rules, player types, and solvers.

Both games share one board: 100 soldiers spread over castles worth 1..5 points.
Game 1 scores an allocation head to head against every other applicant.
Game 2 drops it into random groups of four, ranked by castle points, paying 9/4/2/0.
"""

import functools
import itertools
import math
from typing import NamedTuple

import numpy as np

VALUES = np.array([1, 2, 3, 4, 5], dtype=np.int64)
N_CASTLES = len(VALUES)
TOTAL_SOLDIERS = 100
TOTAL_POINTS = int(VALUES.sum())  # 15

GROUP_SIZE = 4
PAYOUTS = np.array([9.0, 4.0, 2.0, 0.0])

# Game 2 castle points are kept in twelfths so castles split 2, 3 or 4 ways stay
# exact integers. Ranking compares totals for equality, and float thirds would
# turn genuine ties into arbitrary 1st/2nd orderings.
SCALE = 12
_PAYOUT_CUMSUM = np.concatenate([[0.0], np.cumsum(PAYOUTS)])


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------

def validate(allocs):
    """Return `allocs` as an int64 array, raising if any allocation is illegal.

    Accepts a single allocation (shape (5,)) or any batch (shape (..., 5)).
    """
    a = np.asarray(allocs)
    if a.ndim == 0 or a.shape[-1] != N_CASTLES:
        raise ValueError(f"allocation must have {N_CASTLES} entries, got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        if not np.issubdtype(a.dtype, np.floating) or not np.all(a == np.round(a)):
            raise ValueError("soldier counts must be integers")
    a = a.astype(np.int64)
    if (a < 0).any():
        raise ValueError("soldier counts must be non-negative")
    if (a.sum(axis=-1) != TOTAL_SOLDIERS).any():
        raise ValueError(f"each allocation must sum to {TOTAL_SOLDIERS}")
    return a


def game1_points(mine, opponents):
    """Castle points `mine` earns against each opponent in Game 1 (ties split).

    Returns a float array with one entry per opponent, each in [0, 15].
    """
    mine = validate(mine)
    opp = validate(opponents).reshape(-1, N_CASTLES)
    share = (mine > opp) + 0.5 * (mine == opp)
    return share @ VALUES


def game1_score(mine, field):
    """Game 1 score: average castle points against every allocation in `field`."""
    return float(game1_points(mine, field).mean())


def game2_castle_points(groups):
    """Castle points per player in each Game 2 group, in twelfths of a point.

    groups: int array of shape (..., GROUP_SIZE, 5). Returns shape (..., GROUP_SIZE).
    Every castle is awarded, even when all players send zero: a four-way tie at
    zero splits it four ways like any other tie.
    """
    g = np.asarray(groups)
    if g.ndim < 2 or g.shape[-2:] != (GROUP_SIZE, N_CASTLES):
        raise ValueError(f"groups must have shape (..., {GROUP_SIZE}, {N_CASTLES})")
    top = g.max(axis=-2, keepdims=True)
    winners = g == top
    n_winners = winners.sum(axis=-2, keepdims=True)
    # 12 * value is divisible by any winner count in 1..4, so this is exact.
    return (winners * (SCALE * VALUES) // n_winners).sum(axis=-1)


def rank_payouts(points):
    """Payout per player given each player's total points, shape (..., GROUP_SIZE).

    Tied players split the payouts of the places they jointly occupy: a player
    with k opponents strictly ahead and t players (itself included) on its score
    gets mean(PAYOUTS[k : k + t]).
    """
    p = np.asarray(points)
    me, other = p[..., :, None], p[..., None, :]
    ahead = (other > me).sum(axis=-1)
    tied = (other == me).sum(axis=-1)
    return (_PAYOUT_CUMSUM[ahead + tied] - _PAYOUT_CUMSUM[ahead]) / tied


def game2_payouts(groups):
    """Payout per player in each Game 2 group, shape (..., GROUP_SIZE)."""
    return rank_payouts(game2_castle_points(groups))


# ---------------------------------------------------------------------------
# Allocation space
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def all_allocations():
    """Every legal allocation, C(104, 4) = 4,598,126 rows, for exhaustive checks.

    Stars and bars: 4 bars among 104 slots split the other 100 slots into 5 castles.
    """
    slots = TOTAL_SOLDIERS + N_CASTLES - 1
    bars = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(slots), N_CASTLES - 1)),
        dtype=np.int16,
        count=math.comb(slots, N_CASTLES - 1) * (N_CASTLES - 1),
    ).reshape(-1, N_CASTLES - 1)
    edges = np.pad(bars, ((0, 0), (1, 1)), constant_values=(-1, slots))
    out = (np.diff(edges, axis=1) - 1).astype(np.int8)
    out.flags.writeable = False  # cached and shared between callers
    return out


def round_to_total(shares, step, total=TOTAL_SOLDIERS):
    """Round rows of shares (each summing to 1) to multiples of `step` adding up to `total`.

    Largest-remainder rounding, so the result is always a legal allocation.
    `total` may be a scalar or one value per row; it must be a multiple of `step`.
    """
    shares = np.atleast_2d(shares)
    units = np.broadcast_to(np.asarray(total) // step, (len(shares),))
    raw = shares * units[:, None]
    base = np.floor(raw).astype(np.int64)
    short = units - base.sum(axis=1)
    remainder_rank = np.argsort(np.argsort(base - raw, axis=1), axis=1)
    return (base + (remainder_rank < short[:, None])) * step


# ---------------------------------------------------------------------------
# Game 1 analysis
#
# Fact 1: Game 1 is separable. Against a field, sending s soldiers to castle i is
# worth w[i, s] = v_i * (P(opponent sends < s) + 0.5 * P(opponent sends s)) in
# expectation, and an allocation scores sum_i w[i, a_i]. Only each castle's
# marginal distribution in the field matters, never how opponents correlate
# castles. So a whole field reduces to a (5, 101) table of marginals.
# ---------------------------------------------------------------------------

def marginals(allocs, weights=None):
    """Per-castle distribution of soldier counts in a field, shape (5, 101)."""
    a = np.asarray(allocs).reshape(-1, N_CASTLES)
    weights = np.ones(len(a)) if weights is None else np.asarray(weights, dtype=float)
    counts = np.stack([
        np.bincount(a[:, i], weights=weights, minlength=TOTAL_SOLDIERS + 1)
        for i in range(N_CASTLES)
    ])
    return counts / weights.sum()


def win_value(marg):
    """w[i, s]: expected Game 1 points from sending s soldiers to castle i."""
    below = np.cumsum(marg, axis=1) - marg
    return VALUES[:, None] * (below + 0.5 * marg)


def expected_score(w, allocs):
    """Expected Game 1 score of each allocation in `allocs` given a win-value table."""
    a = np.asarray(allocs)
    # One gather per castle: scoring all 4.6M allocations this way never builds
    # the (4.6M, 5) float temporary that fancy-indexing w directly would.
    return sum(w[i][a[..., i]] for i in range(w.shape[0]))


@functools.lru_cache(maxsize=4)
def _knapsack_index(size):
    # rest[t, s] = t - s: soldiers left for the earlier castles when this castle
    # gets s of the t placed so far. Clipped to 0 where s > t and masked out by `ok`.
    t, s = np.ogrid[:size, :size]
    return np.maximum(t - s, 0), s <= t


def best_response(w):
    """Exact maximizer of sum_i w[i, a_i] over allocations; returns (alloc, value).

    Knapsack DP over castles: best[t] is the top value from the castles seen so
    far using exactly t soldiers. Works for any table shape (castles, soldiers + 1),
    which is what lets the tests check it against brute force on small boards.
    """
    n_castles, size = w.shape
    rest, ok = _knapsack_index(size)
    best = w[0]
    picks = []
    for row in w[1:]:
        table = np.where(ok, best[rest] + row, -np.inf)
        pick = table.argmax(axis=1)
        best = table[np.arange(size), pick]
        picks.append(pick)
    alloc = np.empty(n_castles, dtype=np.int64)
    left = size - 1
    for i in range(n_castles - 1, 0, -1):
        alloc[i] = picks[i - 1][left]
        left -= alloc[i]
    alloc[0] = left
    return alloc, float(best[-1])


def _logsumexp(x, axis=-1):
    top = np.max(x, axis=axis, keepdims=True)
    return np.squeeze(top, axis) + np.log(np.sum(np.exp(x - top), axis=axis))


def _log_partition(logits):
    """Same DP as best_response with (max, +) swapped for (logsumexp, +).

    out[k][t] = log of sum over placements of exactly t soldiers on castles 0..k
    of exp(sum of their logits).
    """
    rest, ok = _knapsack_index(logits.shape[1])
    out = [logits[0]]
    for row in logits[1:]:
        out.append(_logsumexp(np.where(ok, out[-1][rest] + row, -np.inf)))
    return out


def logit_marginals(w, lam):
    """Exact per-castle marginals of the noisy best response p(a) ∝ exp(lam * score(a)).

    lam is an inverse temperature in 1/points: lam = 0 is uniform over all
    allocations, lam -> inf approaches the exact best response.
    """
    logits = lam * np.asarray(w, dtype=float)
    out = np.empty_like(logits)
    for i in range(len(logits)):
        others = _log_partition(np.delete(logits, i, axis=0))[-1]
        # Sending s here leaves exactly (soldiers - s) for the other castles.
        log_p = logits[i] + others[::-1]
        out[i] = np.exp(log_p - _logsumexp(log_p))
    return out


def logit_sample(w, lam, n, rng):
    """Draw n allocations from the noisy best response p(a) ∝ exp(lam * score(a)).

    Exact, without enumerating allocations: run the log-partition DP forward,
    then sample castles backward, each conditional on the soldiers still unplaced.
    """
    logits = lam * np.asarray(w, dtype=float)
    n_castles, size = logits.shape
    rest, ok = _knapsack_index(size)
    log_z = _log_partition(logits)
    out = np.empty((n, n_castles), dtype=np.int64)
    left = np.full(n, size - 1)
    for i in range(n_castles - 1, 0, -1):
        # Row t: P(castle i gets s | t soldiers left for castles 0..i).
        log_p = np.where(ok, logits[i] + log_z[i - 1][rest], -np.inf)
        p = np.exp(log_p - _logsumexp(log_p)[:, None])
        cdf = np.where(ok, np.cumsum(p, axis=1), 1.0)
        # Inverse-CDF sampling for every row at once: offsetting row t by 2t keeps
        # the flattened table sorted, so one searchsorted serves all samples.
        keys = (cdf + 2 * np.arange(size)[:, None]).ravel()
        u = rng.random(n)
        out[:, i] = np.searchsorted(keys, u + 2 * left, side="right") - left * size
        left = left - out[:, i]
    out[:, 0] = left
    return out


def noisy_response(w, lam):
    """Marginals of a (noisy) best response; lam = inf means the exact best response."""
    if np.isinf(lam):
        return marginals(best_response(w)[0])
    return logit_marginals(w, lam)


class FictitiousPlay(NamedTuple):
    marginals: np.ndarray       # (5, 101) marginals of the average strategy
    plays: np.ndarray           # (n_iter + 1, 5) the start plus every best response
    exploitability: np.ndarray  # (n_iter,) best-response gain over 7.5 at each round


def fictitious_play(n_iter, start=(20, 20, 20, 20, 20)):
    """Symmetric fictitious play for Game 1 (fact 2).

    Each round best-responds to the average of all earlier rounds. Game 1 is
    symmetric and constant-sum (value 7.5 each), so the average is an
    epsilon-equilibrium with epsilon = (best response score - 7.5), and
    Robinson (1951) guarantees epsilon -> 0. Only marginals are needed per round.
    """
    plays = [validate(start)]
    marg = marginals(plays[0])
    gaps = np.empty(n_iter)
    for t in range(n_iter):
        alloc, value = best_response(win_value(marg))
        gaps[t] = value - TOTAL_POINTS / 2
        plays.append(alloc)
        marg += (marginals(alloc) - marg) / (t + 2)
    return FictitiousPlay(marg, np.array(plays), gaps)


def mean_allocation(marg):
    """Average soldiers per castle under `marg`, rounded to a legal allocation."""
    mean = marg @ np.arange(marg.shape[1])
    return round_to_total(mean / mean.sum(), 1)[0]


# ---------------------------------------------------------------------------
# Player types
#
# Level 0 players follow habits, not reasoning. Strategic players (cognitive
# hierarchy levels, meta exploiters) play a noisy best response to what they
# believe the field does. Equilibrium players sample fictitious play's average.
# ---------------------------------------------------------------------------

EQUAL_SPLIT = np.array([20, 20, 20, 20, 20])
PROPORTIONAL = np.array([7, 13, 20, 27, 33])  # 100 * v / 15, rounded


def level0_round_numbers(n, rng):
    """A loose 'more for bigger castles' split, rounded to multiples of 5 or 10.

    Mean shares 10/15/20/25/30%; the Dirichlet concentration (12) gives each castle a
    spread of roughly +-10 soldiers, so outputs range from 20-20-20-20-20 to 0-0-20-30-50.
    """
    shares = rng.dirichlet(12 * np.array([0.10, 0.15, 0.20, 0.25, 0.30]), size=n)
    tens = rng.random(n) < 0.5
    return np.where(tens[:, None], round_to_total(shares, 10), round_to_total(shares, 5))


def level0_stack_five(n, rng):
    """'Stack the 5': 40, 50 or 60 on castle 5, the rest roughly by value in 5s."""
    top = rng.choice([40, 50, 60], p=[0.4, 0.4, 0.2], size=n)
    rest = round_to_total(rng.dirichlet(4 * VALUES[:4] / 10, size=n), 5, TOTAL_SOLDIERS - top)
    return np.column_stack([rest, top])


def level0_uniform(n, rng):
    """Uniform over all 4.6M allocations (the textbook level-0 assumption)."""
    return logit_sample(np.zeros((N_CASTLES, TOTAL_SOLDIERS + 1)), 0.0, n, rng)


LEVEL0_HABITS = {
    "equal": lambda n, rng: np.tile(EQUAL_SPLIT, (n, 1)),
    "proportional": lambda n, rng: np.tile(PROPORTIONAL, (n, 1)),
    "round": level0_round_numbers,
    "stack5": level0_stack_five,
    "uniform": level0_uniform,
}

# Two level-0 populations: humans with round-number habits, and the textbook
# uniform-random level 0. Shares in "focal" are judgment calls (see README).
LEVEL0_FLAVORS = {
    "focal": {"equal": 0.15, "proportional": 0.15, "round": 0.45, "stack5": 0.15, "uniform": 0.10},
    "random": {"uniform": 1.0},
}


def level0_sample(flavor, n, rng):
    """n allocations from a level-0 flavor, habits mixed in their listed shares."""
    habits = LEVEL0_FLAVORS[flavor]
    counts = rng.multinomial(n, list(habits.values()))
    out = np.concatenate([LEVEL0_HABITS[h](k, rng) for h, k in zip(habits, counts)])
    return out[rng.permutation(n)]


def poisson_levels(tau, max_level):
    """Shares of thinking levels 0..max_level, Poisson(tau), tail pooled into the top level."""
    k = np.arange(max_level + 1)
    shares = np.exp(-tau) * tau**k / np.array([math.factorial(i) for i in k])
    shares[-1] += 1 - shares.sum()
    return shares


def cognitive_hierarchy(level0_marg, tau, lam, max_level):
    """Game 1 marginals of levels 0..max_level, shape (max_level + 1, 5, 101).

    Level k noisily best-responds to levels 0..k-1 mixed in their Poisson(tau)
    proportions (Camerer, Ho and Chong 2004). The top level's share pools the tail,
    but its beliefs use only the true Poisson weights of the levels below it.
    """
    shares = poisson_levels(tau, max_level)
    levels = [level0_marg]
    for k in range(1, max_level + 1):
        belief = np.tensordot(shares[:k] / shares[:k].sum(), np.array(levels), axes=1)
        levels.append(noisy_response(win_value(belief), lam))
    return np.array(levels)


# ---------------------------------------------------------------------------
# Game 2 simulator
#
# Fact 3: castle points always total 15, so a player with more than 7.5 points
# finishes outright 1st: the other three share less than 7.5. Sweeping any castle
# set worth 8 or more ({5,3}, {5,4}, {5,2,1}, {4,3,1}, {4,3,2} are the minimal
# ones) therefore guarantees the 9-point payout, whatever happens elsewhere.
# ---------------------------------------------------------------------------

class Field(NamedTuple):
    """A population of players as a discrete distribution over allocations."""
    allocs: np.ndarray  # (N, 5)
    probs: np.ndarray   # (N,), sums to 1


def field_from_samples(allocs):
    allocs = validate(allocs).reshape(-1, N_CASTLES)
    return Field(allocs, np.full(len(allocs), 1 / len(allocs)))


def mix_fields(parts):
    """Mixture of fields from (weight, Field) pairs; weights are renormalized."""
    parts = [(w, f) for w, f in parts if w > 0]
    total = sum(w for w, _ in parts)
    return Field(np.concatenate([f.allocs for _, f in parts]),
                 np.concatenate([w / total * f.probs for w, f in parts]))


def field_marginals(field):
    return marginals(field.allocs, field.probs)


def sample_opponents(field, n_groups, rng):
    """The other three players in each of n_groups random groups, shape (n_groups, 3, 5)."""
    idx = rng.choice(len(field.probs), size=(n_groups, GROUP_SIZE - 1), p=field.probs)
    return field.allocs[idx]


class Opponents(NamedTuple):
    """Fixed opponent groups, preprocessed so a candidate is scored without re-ranking."""
    top: np.ndarray         # (G, 5) most soldiers any opponent sends to each castle
    at_top: np.ndarray      # (G, 3, 5) which opponents send that many
    lose_share: np.ndarray  # (G, 5) twelfths each top opponent gets if I send fewer
    tie_share: np.ndarray   # (G, 5) twelfths each top opponent (and I) get if I tie the top


def face(opponent_allocs):
    o = np.asarray(opponent_allocs)
    top = o.max(axis=1)
    at_top = o == top[:, None, :]
    n_top = at_top.sum(axis=1)
    v = SCALE * VALUES
    return Opponents(top, at_top.astype(np.int32), (v // n_top).astype(np.int32),
                     (v // (n_top + 1)).astype(np.int32))


def payouts(candidates, opp, max_cells=2_000_000):
    """Payout of each candidate in each fixed group, shape (B, G).

    Every candidate faces the same groups (common random numbers), so differences
    between candidates carry far less noise than their separate averages.
    """
    cands = np.atleast_2d(candidates)
    n_groups = len(opp.top)
    chunk = max(1, max_cells // n_groups)
    out = np.empty((len(cands), n_groups))
    v = (SCALE * VALUES).astype(np.int32)
    for start in range(0, len(cands), chunk):
        a = cands[start:start + chunk, None, :]
        win = a > opp.top
        tie = a == opp.top
        mine = (win * v + tie * opp.tie_share).sum(axis=-1)
        award = np.where(win, 0, np.where(tie, opp.tie_share, opp.lose_share))
        theirs = np.stack([(award * opp.at_top[:, j]).sum(axis=-1) for j in range(GROUP_SIZE - 1)], axis=-1)
        ahead = (theirs > mine[..., None]).sum(axis=-1)
        tied = 1 + (theirs == mine[..., None]).sum(axis=-1)
        out[start:start + chunk] = (_PAYOUT_CUMSUM[ahead + tied] - _PAYOUT_CUMSUM[ahead]) / tied
    return out


def castle_share_4way(marg):
    """q[i, s]: expected share of castle i won by sending s against 3 opponents from the field.

    Exact for one castle, since the three opponents are independent players:
    j of them tie my s and the rest send less with probability C(3, j) p^j F^(3-j).
    """
    below = np.cumsum(marg, axis=1) - marg
    return sum(math.comb(3, j) * marg**j * below ** (3 - j) / (1 + j) for j in range(4))


def sweep_plan(marg, castle_set):
    """Allocation with the best chance of sweeping `castle_set` (castle numbers 1-5).

    The DP maximizes the sum over the set of log q_i(a_i), which is the chance of
    winning every castle in it if castles were independent. Leftover soldiers go to
    expected points elsewhere, at a weight too small to ever pull them out of the set.
    """
    q = castle_share_4way(marg)
    in_set = np.isin(np.arange(1, N_CASTLES + 1), list(castle_set))
    w = np.where(in_set[:, None], np.log(np.maximum(q, 1e-12)), 1e-3 * VALUES[:, None] * q)
    return best_response(w)[0]


CASTLE_SETS = [s for r in range(1, N_CASTLES + 1)
               for s in itertools.combinations(range(1, N_CASTLES + 1), r)]


def game2_plans(marg):
    """Seed allocations for the Game 2 best-response search, tuned to a field.

    A sweep plan for each of the 31 castle subsets, plus the plan maximizing
    expected castle points against 3 opponents and the Game 1 best response.
    """
    plans = [best_response(VALUES[:, None] * castle_share_4way(marg))[0],
             best_response(win_value(marg))[0]]
    plans += [sweep_plan(marg, s) for s in CASTLE_SETS]
    return np.unique(plans, axis=0)


# Every way to move 1-5 soldiers from one castle to another.
_MOVES = np.array([
    d * (np.eye(N_CASTLES, dtype=np.int64)[j] - np.eye(N_CASTLES, dtype=np.int64)[i])
    for i in range(N_CASTLES) for j in range(N_CASTLES) if i != j for d in range(1, 6)
])


def local_search(start, opp, max_steps=1000):
    """Steepest ascent on mean payout against fixed groups, moving 1-5 soldiers at a time."""
    current = validate(start)
    value = payouts(current, opp).mean()
    for _ in range(max_steps):
        neighbors = current + _MOVES
        neighbors = neighbors[(neighbors >= 0).all(axis=1)]
        values = payouts(neighbors, opp).mean(axis=1)
        k = values.argmax()
        if values[k] <= value:
            break
        current, value = neighbors[k], values[k]
    return current, float(value)


def anneal(start, opp, steps, rng, temp=0.05):
    """Simulated annealing on mean payout against fixed groups; returns the best seen.

    Temperature falls linearly from `temp` (payout units) to zero, so early on it
    accepts moves that lose a little in order to escape local optima.
    """
    current = validate(start)
    value = payouts(current, opp).mean()
    best, best_value = current, value
    for step in range(steps):
        cand = current + _MOVES[rng.integers(len(_MOVES))]
        if (cand < 0).any():
            continue
        cand_value = payouts(cand, opp).mean()
        heat = temp * (1 - step / steps)
        if cand_value >= value or rng.random() < np.exp((cand_value - value) / max(heat, 1e-12)):
            current, value = cand, cand_value
            if value > best_value:
                best, best_value = current, value
    return best, float(best_value)


class Game2Response(NamedTuple):
    plans: np.ndarray   # (P, 5) distinct plans considered
    values: np.ndarray  # (P,) mean payout of each against the target field


def game2_response(field, rng, n_groups=4000, polish=True):
    """Approximate Game 2 best response to a field.

    No DP exists here (the ranking couples castles), so DP-seeded plans are scored
    by simulation and the best one is polished by local search.
    """
    opp = face(sample_opponents(field, n_groups, rng))
    plans = game2_plans(field_marginals(field))
    values = payouts(plans, opp).mean(axis=1)
    if polish:
        k = values.argmax()
        plans[k], values[k] = local_search(plans[k], opp)
        plans, keep = np.unique(plans, axis=0, return_index=True)
        values = values[keep]
    return Game2Response(plans, values)


def logit_field(response, lam):
    """Noisy Game 2 best response: logit over the considered plans (lam in 1/payout)."""
    if np.isinf(lam):
        return Field(response.plans[[response.values.argmax()]], np.ones(1))
    p = np.exp(lam * (response.values - response.values.max()))
    return Field(response.plans, p / p.sum())


def replicator(pool, rng, steps=300, n_groups=3000, eta=2.0, mutation=0.002):
    """Noisy replicator dynamics over a pool of Game 2 allocations.

    Each step re-weights the pool by exp(eta * payoff advantage) against groups
    drawn from the current mix, then mixes in a little uniform noise so no plan
    dies out for good. Returns the time-averaged mix, an approximate symmetric
    equilibrium; its quality is checked by its exploitability, not assumed.
    """
    x = np.full(len(pool), 1 / len(pool))
    avg = np.zeros_like(x)
    for _ in range(steps):
        opp = face(sample_opponents(Field(pool, x), n_groups, rng))
        u = payouts(pool, opp).mean(axis=1)
        x = x * np.exp(eta * (u - u @ x))
        x = (1 - mutation) * x / x.sum() + mutation / len(pool)
        avg += x
    return Field(pool, avg / steps)


def fictitious_play_game2(n_iter, rng, n_groups=2000, checkpoints=()):
    """Symmetric fictitious play for Game 2 with the search-based best response.

    Each round adds an approximate best response to the uniform mix of all earlier
    plays. Returns the plays and, at each checkpoint round, the exploitability of
    the mix so far (best-response payout minus the symmetric value 3.75), measured
    on fresh groups. The oracle is approximate, so that is a lower bound.
    """
    plays = [EQUAL_SPLIT.copy()]
    gaps = {}
    for t in range(1, n_iter + 1):
        r = game2_response(field_from_samples(np.array(plays)), rng, n_groups)
        plays.append(r.plans[r.values.argmax()])
        if t in checkpoints:
            check = game2_response(field_from_samples(np.array(plays)), rng, 3 * n_groups)
            gaps[t] = check.values.max() - TOTAL_POINTS / GROUP_SIZE
    return np.array(plays), gaps
