"""Infer the Game 1 field from the public leaderboard, then best-respond to it.

The board lists every entry's score but no allocations. In Game 1 an entry's score is
sum_i w[i, a_i] with w = win_value(field marginals), and the field is the board itself.
So the board is a mixture model whose component locations depend on its own weights.
This module fits it with a fixed-point EM over a library of candidate allocations:

  E-step  each board score is explained by library allocations whose predicted score is
          close to it, in proportion to their current weight.
  M-step  weights become the expected counts (plus prior pseudo-counts), and every
          predicted score is recomputed against the new field.

The noise width sigma is annealed down so that copy spikes (dozens of identical scores)
lock onto one allocation once the field is roughly right. Known (allocation, score)
pairs are enforced exactly by an exponential tilt of the weights, which is the
smallest change in KL divergence that satisfies them.
"""

import csv
import functools
from pathlib import Path
from typing import NamedTuple

import numpy as np

from sim import castles as c
from sim.experiments import GAME1_SEEDS

# Splits published outside the board (posts, READMEs). Newcomers can copy them, so they are
# focal points in the prior and copy targets in the late-mover worlds.
KNOWN_SPLITS = {"own Boston entry": [8, 17, 30, 44, 1],
                "posted Boston leader": [5, 7, 35, 47, 6],
                "Nischal (GitHub)": [1, 0, 21, 31, 47],
                "Lachlan (GitHub)": [5, 16, 26, 1, 52],
                "qbeka (GitHub)": [3, 5, 27, 2, 63],
                "Mianmalol (GitHub)": [3, 5, 4, 43, 45],
                "HaykGar (GitHub)": [1, 1, 1, 41, 56],
                "notnoobster (GitHub)": [4, 4, 4, 44, 44],
                "blobberus (GitHub)": [2, 16, 1, 38, 43],
                "Aden Nowell (GitHub)": [10, 16, 25, 45, 4],
                "most common chatbot answer": [7, 13, 20, 27, 33]}

# Real chatbot answers to the challenge prompt: 150 Claude chats collected by
# blobberus/CASTLE plus six GPT/Gemini/other baselines from AdenNowell/Allocation-Study-Sequoia.
# Kept in data/ (gitignored) since it is someone else's dataset; the model runs without it.
CHATBOT_FILE = Path(__file__).resolve().parents[1] / "data" / "external" / "chatbot_answers.csv"


@functools.lru_cache(maxsize=2)
def chatbot_answers(game="game1"):
    """One row per chat (duplicates kept, so frequencies carry over); empty if missing."""
    if not CHATBOT_FILE.exists():
        return np.empty((0, c.N_CASTLES), dtype=np.int64)
    with open(CHATBOT_FILE, newline="") as f:
        rows = [r[game] for r in csv.DictReader(f) if r[game]]
    return np.array([[int(x) for x in r.split("-")] for r in rows], dtype=np.int64)

# Prior mass per player source. Judgment calls, same spirit as PRIORS in experiments.py:
# the board then reweights within and across sources.
SOURCE_MASS = {
    "chatbot": 0.10,      # pasted a chatbot answer (real answers, data/external)
    "habit": 0.20,        # round numbers, equal split, proportional, stack the 5
    "lattice": 0.05,      # floor on every multiple-of-5 split, so any round copy spike is reachable
    "focal": 0.05,        # named seeds and posted splits
    "levelk": 0.25,       # cognitive hierarchy above the habit players
    "equilibrium": 0.10,  # fictitious-play equilibrium plays
    "sharp": 0.30,        # noisy best responses to the fitted field itself (probers, modelers)
}
SHARP_LAMS = (8.0, 20.0, 60.0)  # inverse temperatures (1/points): sloppy to near-exact


class Library(NamedTuple):
    allocs: np.ndarray  # (M, 5) distinct candidate allocations
    prior: np.ndarray   # (M,) prior weight, sums to 1
    source: np.ndarray  # (M,) source holding the most prior mass for that allocation


class Fit(NamedTuple):
    weights: np.ndarray  # (M,) fitted share of the field on each library allocation
    scores: np.ndarray   # (M,) each allocation's Game 1 score against the fitted field
    marg: np.ndarray     # (5, 101) fitted field marginals
    loglik: float        # board log-likelihood at the final sigma


# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------

def level_k_samples(level0_marg, tau, lam, n, rng, max_level=6):
    """n allocations from cognitive-hierarchy levels 1..max_level in Poisson(tau) shares."""
    shares = c.poisson_levels(tau, max_level)
    counts = rng.multinomial(n, shares[1:] / shares[1:].sum())
    levels, out = [level0_marg], []
    for k in range(1, max_level + 1):
        belief = np.tensordot(shares[:k] / shares[:k].sum(), np.array(levels), axes=1)
        w = c.win_value(belief)
        out.append(c.logit_sample(w, lam, counts[k - 1], rng))
        levels.append(c.logit_marginals(w, lam))
    return np.concatenate(out)


def sharp_samples(marg, n, rng, lams=SHARP_LAMS):
    w = c.win_value(marg)
    return np.concatenate([c.logit_sample(w, lam, n, rng) for lam in lams])


def _pool(parts):
    """Merge (allocs, total mass, source) parts into one deduplicated Library.

    Deduplication matters: a copy spike must sit on one row, not be split across
    identical rows that the EM would have to agree on.
    """
    allocs = np.concatenate([np.asarray(a, dtype=np.int64) for a, _, _ in parts])
    mass = np.concatenate([np.full(len(a), m / len(a)) for a, m, _ in parts])
    src = np.concatenate([np.full(len(a), i) for i, (a, _, _) in enumerate(parts)])
    uniq, inv = np.unique(allocs, axis=0, return_inverse=True)
    inv = inv.ravel()
    prior = np.bincount(inv, weights=mass, minlength=len(uniq))
    # Label each allocation by whichever source gave it the most prior mass.
    by_src = np.zeros((len(uniq), len(parts)))
    np.add.at(by_src, (inv, src), mass)
    names = np.array([s for _, _, s in parts])
    return Library(uniq, prior / prior.sum(), names[by_src.argmax(axis=1)])


@functools.lru_cache(maxsize=1)
def _equilibrium_plays():
    # Deterministic and ~1 s; every ensemble member reuses it.
    return c.fictitious_play(1500).plays[1:]


def build_library(rng, sharp_marg=None, n=4000, mass=SOURCE_MASS, tau=2.5,
                  levelk_lams=(3.0, 8.0), sharp_lams=SHARP_LAMS):
    """Candidate allocations with prior weights. `sharp_marg` is the field the sharp
    players respond to; by default the field implied by the other sources."""
    habit = np.concatenate([c.level0_round_numbers(6 * n, rng), c.level0_stack_five(2 * n, rng),
                            np.tile(c.EQUAL_SPLIT, (n, 1)), np.tile(c.PROPORTIONAL, (n, 1))])
    full = c.all_allocations()
    lattice = full[(full % 5 == 0).all(axis=1)]
    focal = np.array(list(GAME1_SEEDS.values()) + list(KNOWN_SPLITS.values()))
    level0_marg = c.marginals(habit)
    levelk = np.concatenate([level_k_samples(level0_marg, tau, lam, n, rng) for lam in levelk_lams])
    equilibrium = _equilibrium_plays()

    parts = [(habit, mass["habit"], "habit"), (lattice, mass["lattice"], "lattice"),
             (focal, mass["focal"], "focal"), (levelk, mass["levelk"], "levelk"),
             (equilibrium, mass["equilibrium"], "equilibrium")]
    chat = chatbot_answers("game1")
    if len(chat) and mass.get("chatbot", 0) > 0:
        parts.append((chat, mass["chatbot"], "chatbot"))
    if sharp_marg is None:
        base = _pool(parts)
        sharp_marg = c.marginals(base.allocs, base.prior)
    parts.append((sharp_samples(sharp_marg, n, rng, sharp_lams), mass["sharp"], "sharp"))
    return _pool(parts)


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def field_scores(allocs, weights):
    """Game 1 score of every allocation against the field `weights` puts on `allocs`."""
    return c.expected_score(c.win_value(c.marginals(allocs, weights)), allocs)


def payoff_vs(alloc, allocs):
    """Game 1 points `alloc` takes head to head from each row of `allocs`."""
    a = np.asarray(alloc)
    return ((a > allocs) * c.VALUES + 0.5 * (a == allocs) * c.VALUES).sum(axis=1)


def _tilt_one(w, g, target, tol=1e-7):
    """w * exp(theta * g), renormalized, with theta solving E[g] = target.

    E[g] is increasing in theta with derivative Var[g], so safeguarded Newton converges
    in a few steps; it falls back to bisection whenever a step leaves the bracket.
    """
    lo, hi, theta = -60.0, 60.0, 0.0
    for _ in range(60):
        z = theta * g
        t = w * np.exp(z - z.max())  # shift by the max so neither sign of theta overflows
        t /= t.sum()
        mean = t @ g
        f = mean - target
        if abs(f) < tol:
            break
        if f < 0:
            lo = theta
        else:
            hi = theta
        var = t @ (g * g) - mean**2
        step = theta - f / var if var > 1e-12 else np.nan
        theta = step if lo < step < hi else 0.5 * (lo + hi)
    return t


def tilt_to_anchors(w, allocs, anchors, sweeps=3, tol=1e-4):
    """Exponentially tilt w so each known (alloc, score) pair holds.

    A score is linear in the field weights, so tilting by exp(theta * payoff) moves it
    monotonically in theta. Cycling over anchors is iterative I-projection onto their
    intersection (Csiszar); it stops early once every anchor is within `tol`.
    """
    payoffs = [(payoff_vs(alloc, allocs), target) for alloc, target in anchors]
    for _ in range(sweeps):
        for g, target in payoffs:
            w = _tilt_one(w, g, target)
        if all(abs(w @ g - target) < tol for g, target in payoffs):
            break
    return w


def tilt_to_board_top(prior, allocs, anchors, top_score, iters=8, tol=0.003, sweeps=3):
    """Tilt to the anchors plus one more fact: the best achievable score equals the board top.

    The top of the board is a cluster of exact ties (several entries at 9.315), which is
    what independent probers converging on the same exact best response look like. So
    the field's best response should score the board's top score. That pins how
    exploitable the field's bunching is, which the anchors alone leave loose. The best
    response moves as the weights move, so iterate: tilt with the current best response
    as a pseudo-anchor, recompute it, repeat until the maximum matches.
    """
    w = tilt_to_anchors(prior, allocs, anchors, sweeps)
    for _ in range(iters):
        br, value = c.best_response(c.win_value(c.marginals(allocs, w)))
        if abs(value - top_score) < tol:
            break
        w = tilt_to_anchors(prior, allocs, list(anchors) + [(br, top_score)], sweeps)
    return w


def e_step(y, cnt, s, w, sigma, chunk=2048):
    """Expected board count on each library allocation, and the board log-likelihood.

    Streams over library chunks with a running log-sum-exp, so memory stays at
    (distinct scores x chunk) instead of (distinct scores x library).
    """
    logw = np.log(np.maximum(w, 1e-300))
    inv2s2 = 0.5 / sigma**2
    m = np.full(len(y), -np.inf)
    acc = np.zeros(len(y))
    for j in range(0, len(s), chunk):
        L = logw[None, j:j + chunk] - inv2s2 * (y[:, None] - s[None, j:j + chunk]) ** 2
        new_m = np.maximum(m, L.max(axis=1))
        acc = acc * np.exp(m - new_m) + np.exp(L - new_m[:, None]).sum(axis=1)
        m = new_m
    lse = m + np.log(acc)
    counts = np.empty(len(s))
    for j in range(0, len(s), chunk):
        L = logw[None, j:j + chunk] - inv2s2 * (y[:, None] - s[None, j:j + chunk]) ** 2
        counts[j:j + chunk] = cnt @ np.exp(L - lse[:, None])
    loglik = float(cnt @ (lse - 0.5 * np.log(2 * np.pi * sigma**2)))
    return counts, loglik


def fit_board(board_scores, lib, anchors=(), sigmas=(0.08, 0.04, 0.02, 0.01), iters=10,
              prior_strength=0.25, damping=0.5, w0=None):
    """Fixed-point EM for the field weights over `lib` that explain `board_scores`.

    prior_strength is the prior's pseudo-count as a fraction of the board size: it
    keeps weight on allocations the board can't distinguish by score alone.
    """
    y, cnt = np.unique(np.round(np.asarray(board_scores, dtype=float), 3), return_counts=True)
    n = cnt.sum()
    w = lib.prior.copy() if w0 is None else np.asarray(w0, dtype=float).copy()
    w = tilt_to_anchors(w, lib.allocs, anchors)
    loglik = np.nan
    for sigma in sigmas:
        for _ in range(iters):
            s = field_scores(lib.allocs, w)
            counts, loglik = e_step(y, cnt, s, w, sigma)
            target = (counts + prior_strength * n * lib.prior) / (n * (1 + prior_strength))
            # Damped update: scores move with the weights, so a full step can oscillate.
            w = tilt_to_anchors((1 - damping) * w + damping * target, lib.allocs, anchors)
    s = field_scores(lib.allocs, w)
    return Fit(w, s, c.marginals(lib.allocs, w), loglik)


def refit_with_sharp(board_scores, rng, anchors=(), rounds=2, **kw):
    """Fit, then rebuild the 'sharp' players as noisy best responses to the fitted field
    and fit again. Sharp players respond to the real field, not to my prior about it."""
    lib = build_library(rng)
    fit = fit_board(board_scores, lib, anchors, **kw)
    for _ in range(rounds - 1):
        lib = build_library(rng, sharp_marg=fit.marg)
        fit = fit_board(board_scores, lib, anchors, **kw)
    return lib, fit


# ---------------------------------------------------------------------------
# Synthetic check: how much does a scores-only board reveal?
# ---------------------------------------------------------------------------

def synthetic_board(rng, n=5840):
    """A field built from assumptions the library does NOT share, plus copy spikes.

    Returns (entries (n, 5), board scores rounded like the site's).
    """
    habit = c.level0_sample("focal", n, rng)
    level0_marg = c.marginals(habit)
    shallow = level_k_samples(level0_marg, 1.5, 3.0, n, rng)          # library uses tau 2.5
    eq = c.fictitious_play(800).plays[1:]
    crowd = np.concatenate([habit[: int(0.45 * n)], shallow[: int(0.32 * n)],
                            eq[rng.integers(0, len(eq), int(0.1 * n))]])
    probers = sharp_samples(c.marginals(crowd), int(0.1 * n), rng, lams=(30.0,))
    spikes = np.repeat(np.array([[5, 7, 35, 47, 6], [3, 11, 27, 3, 56], [0, 0, 30, 30, 40],
                                 [2, 4, 24, 34, 36], [10, 10, 25, 25, 30]]),
                       [40, 30, 35, 25, 30], axis=0)
    entries = np.concatenate([crowd, probers, spikes])[:n]
    scores = field_scores(entries, np.ones(len(entries)))
    return entries, np.round(scores, 3)


def regret(true_marg, alloc):
    """Points lost by playing `alloc` instead of the best response to the true field."""
    w = c.win_value(true_marg)
    return c.best_response(w)[1] - c.expected_score(w, np.asarray(alloc)[None])[0]
