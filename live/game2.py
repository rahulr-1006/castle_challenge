"""Game 2: fit the field from anchors, then pick for #1 against hidden late trackers.

Step 1, noise. Every hourly board re-draws each entry's groups, so successive snapshots
give independent reads of an anchor's score. Average over a trailing window and report
the read-to-read SD as the measured noise.

Step 2, populations. The Game 2 field is a mixture of
  A  pasters: people who reused their Game 1 answer (the anchored Game 1 field),
  B  optimizers: noisy Game 2 best responses to the pasters, and one level above that,
  C  copiers: published Game 2 splits and the all-on-castle-5 habit,
  D  equilibrium players: fictitious-play Game 2 plays (results/simulation/game2_equilibrium_plays.csv),
  E  chatbot pasters: real chatbot answers to the Game 2 prompt, at their real frequencies
     (data/external/chatbot_answers.csv; 0-0-38-0-62 and 0-0-36-0-64 are the favorites).
The first fit used A-C only and failed its own check: the best payout achievable against
it was 7.24 while the real board top is 5.84, so the real field is far harder to
exploit. D adds that hardness, and the fit now also matches the board top: weights are
chosen to fit the averaged anchors *and* make the best achievable payout equal the
board's top score, the same constraint that pins Game 1. Every grid point is scored on
the same simulated groups (common random numbers).

Step 3, pick. Hidden late entrants include trackers who fit the same board and
best-respond (L1), or best-respond to those (L2). A candidate wins a world if its payout
beats every rival's on the same groups.
"""

from pathlib import Path
from typing import NamedTuple

import itertools

import numpy as np
import pandas as pd

from live import board_fit as bf
from sim import castles as c
from live import probes
from live import scrape_leaderboard as sl
from live import strategy as st

PUBLIC_G2 = {"own Boston entry": [18, 8, 2, 2, 70],
             "HaykGar (GitHub)": [0, 1, 42, 56, 1],
             "notnoobster (GitHub)": [10, 41, 48, 0, 1],
             "blobberus (GitHub)": [1, 37, 0, 62, 0],
             "Nischal (GitHub)": [0, 0, 48, 52, 0],
             "Lachlan (GitHub)": [12, 17, 0, 0, 71],
             "qbeka (GitHub)": [2, 34, 3, 59, 2],
             "Mianmalol (GitHub)": [0, 36, 0, 64, 0],
             "all on castle 5": [0, 0, 0, 0, 100]}
POPULATIONS = ("pasters", "optimizers", "copiers", "equilibrium", "chatbot")
EQ_PLAYS = Path(__file__).resolve().parents[1] / "results" / "simulation" / "game2_equilibrium_plays.csv"
ANCHOR_SD = 0.15  # model error per anchor (points); measured board noise is ~0.005
TOP_SD = 0.10     # how closely the best achievable payout must match the board top
WINDOW = 6
LATE = {"count": (100, 600), "sharp_share": (0.20, 0.60), "lam": 20.0}


class AveragedAnchor(NamedTuple):
    label: str
    alloc: np.ndarray
    mean: float
    sd: float  # read-to-read SD across snapshots (NaN with one read)
    reads: int


class G2Fit(NamedTuple):
    weights: np.ndarray        # share of each of POPULATIONS
    field: c.Field
    pred: np.ndarray           # model score of each anchor at the best weights
    grid: pd.DataFrame         # every grid point with its squared error
    board_top_pred: float      # best Game 2 payout achievable against the fitted field


def averaged_anchors(window=WINDOW, files=None):
    files = sl.snapshot_files() if files is None else files
    reads = {}
    # Distinct boards only: a republished board repeats the same simulation draw.
    for board in probes.distinct_boards("g2", files, window):
        for label, alloc, score in st.load_anchors(board, game="g2"):
            reads.setdefault((label, tuple(alloc)), []).append(score)
    return [AveragedAnchor(label, np.array(alloc), float(np.mean(v)),
                           float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"), len(v))
            for (label, alloc), v in reads.items()]


def pasters(g1_board, rng):
    """The anchored Game 1 field: what people who reused their Game 1 answer play."""
    anchors = [(a, s) for _, a, s in st.load_anchors(g1_board)]
    lib = bf.build_library(rng)
    w = bf.tilt_to_board_top(lib.prior, lib.allocs, anchors, float(g1_board["p"].max()), sweeps=50)
    keep = w > 1e-7  # drop negligible rows so group sampling stays fast
    return c.Field(lib.allocs[keep], w[keep] / w[keep].sum())


def optimizers(pasted, rng, lams=(1.0, 3.0), n_groups=4000):
    """Level 1: noisy best responses to the pasters. Level 2: to pasters plus level 1."""
    l1 = c.mix_fields([(1.0, c.logit_field(c.game2_response(pasted, rng, n_groups), lam))
                       for lam in lams])
    belief = c.mix_fields([(0.5, pasted), (0.5, l1)])
    l2 = c.mix_fields([(1.0, c.logit_field(c.game2_response(belief, rng, n_groups), lam))
                       for lam in lams])
    return c.mix_fields([(0.5, l1), (0.5, l2)])


def copiers(public=PUBLIC_G2):
    return c.field_from_samples(np.array(list(public.values())))


def equilibrium_players(burn_in=100):
    """Fictitious-play Game 2 plays; their running mix approximates the symmetric equilibrium."""
    plays = pd.read_csv(EQ_PLAYS).to_numpy()[burn_in:]
    return c.field_from_samples(plays)


def chatbot_pasters():
    """Real chatbot Game 2 answers, one row per chat so their frequencies carry over."""
    chat = bf.chatbot_answers("game2")
    return c.field_from_samples(chat) if len(chat) else copiers()


def simplex_grid(step, k):
    """Every weight vector over k populations on a grid of `step`."""
    n = int(round(1 / step))
    pts = [p for p in itertools.product(range(n + 1), repeat=k - 1) if sum(p) <= n]
    return np.array([(*p, n - sum(p)) for p in pts]) / n


def mixed_opponents(pools, u, weights):
    """Opponents for a mixture, reusing fixed per-population draws and fixed uniforms."""
    which = (u[..., None] >= np.cumsum(weights)[:-1]).sum(axis=-1)  # (G, 3) population index
    g, k = np.indices(u.shape)
    return np.stack(pools)[which, g, k]


def strong_plans(pops, rng):
    """Plans that can score near the top against mixtures of these populations, used to
    estimate each grid point's best achievable payout without a full search per point."""
    plans = [np.array(list(PUBLIC_G2.values()))]
    for p in pops:
        plans.append(c.game2_plans(c.field_marginals(p)))
    even = c.mix_fields([(1.0, p) for p in pops])
    plans.append(c.game2_response(even, rng, n_groups=4000).plans)
    return np.unique(np.concatenate(plans), axis=0)


def fit_mixture(pops, anchors, board_top, rng, n_groups=6_000, step=0.1):
    allocs = np.array([a.alloc for a in anchors])
    obs = np.array([a.mean for a in anchors])
    plans = strong_plans(pops, rng)
    pools = [p.allocs[rng.choice(len(p.probs), size=(n_groups, c.GROUP_SIZE - 1), p=p.probs)]
             for p in pops]
    u = rng.random((n_groups, c.GROUP_SIZE - 1))
    rows, preds = [], []
    for weights in simplex_grid(step, len(pops)):
        opp = c.face(mixed_opponents(pools, u, weights))
        pred = c.payouts(allocs, opp).mean(axis=1)
        top = c.payouts(plans, opp).mean(axis=1).max()
        loss = np.sum(((pred - obs) / ANCHOR_SD) ** 2) + ((top - board_top) / TOP_SD) ** 2
        rows.append((*weights, float(np.sum((pred - obs) ** 2)), float(top), float(loss)))
        preds.append(pred)
    grid = pd.DataFrame(rows, columns=[*POPULATIONS[:len(pops)], "sse", "best_payout", "loss"])
    best = int(grid["loss"].idxmin())
    weights = grid.loc[best, list(POPULATIONS[:len(pops)])].to_numpy(dtype=float)
    field = c.mix_fields(list(zip(weights, pops)))
    top = c.game2_response(field, rng, n_groups=8000).values.max()
    return G2Fit(weights, field, preds[best], grid, float(top))


class G2Pick(NamedTuple):
    plans: np.ndarray
    p_first: np.ndarray
    mean_payout: np.ndarray


def swing_pick(fit, n_existing, rng, n_worlds=40, n_groups=20_000, public=PUBLIC_G2):
    """Rank candidate plans by P(beating every rival) across hidden-late-entrant worlds."""
    r1 = c.game2_response(fit.field, rng, n_groups=6000)
    l1 = c.logit_field(r1, LATE["lam"])
    rivals_top = r1.plans[np.argsort(-r1.values)[:20]]  # stand-ins for the board's leaders
    cands = [r1.plans]
    worlds = []
    for _ in range(n_worlds):
        count = int(rng.integers(*LATE["count"]))
        n_sharp = int(rng.uniform(*LATE["sharp_share"]) * count)
        n_l2 = int(rng.uniform(0, 1) * n_sharp)
        belief = c.mix_fields([(n_existing, fit.field), (n_sharp, l1)])
        r2 = c.game2_response(belief, rng, n_groups=3000, polish=False)
        l2 = c.logit_field(r2, LATE["lam"])
        n_plain = count - n_sharp
        final = c.mix_fields([(n_existing + int(0.8 * n_plain), fit.field),
                              (n_plain - int(0.8 * n_plain), copiers(public)),
                              (n_sharp - n_l2, l1), (n_l2, l2)])
        worlds.append((final, np.concatenate([rivals_top, r2.plans[np.argsort(-r2.values)[:10]]])))
        cands.append(r2.plans)
    pool = np.unique(np.concatenate(cands), axis=0)
    wins, payout = np.zeros(len(pool)), np.zeros(len(pool))
    for final, rivals in worlds:
        opp = c.face(c.sample_opponents(final, n_groups, rng))
        mine = c.payouts(pool, opp).mean(axis=1)
        best_rival = c.payouts(rivals, opp).mean(axis=1).max()
        wins += mine >= best_rival
        payout += mine
    return G2Pick(pool, wins / n_worlds, payout / n_worlds)


def run(rng, window=WINDOW):
    files = sl.snapshot_files()
    data = sl.read_snapshot(files[-1])
    g1, g2 = pd.DataFrame(data["g1"]), pd.DataFrame(data["g2"])
    anchors = averaged_anchors(window, files)
    a = pasters(g1, rng)
    pops = [a, optimizers(a, rng), copiers(), equilibrium_players(), chatbot_pasters()]
    fit = fit_mixture(pops, anchors, float(g2["p"].max()), rng)
    pick = swing_pick(fit, len(g2), rng)
    return data["generated_at"], g2, anchors, fit, pick
