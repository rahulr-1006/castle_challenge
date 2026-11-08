"""Track what new (Pittsburgh) applicants play, using anchor score changes between snapshots.

When k entries join the board, an anchor's head-to-head total rises by exactly its points
against them:
    (N_t - 1) * s_t  -  (N_0 - 1) * s_0  =  sum over newcomers of payoff(anchor, newcomer)
so every anchor reads off the newcomers' average result against a known split. The site
rounds scores to 3 dp, which puts about +-N * 0.0005 on each total; the error on the
cohort average shrinks as 1/k, so the widest window (first snapshot vs latest) is the
sharpest read.

Those cohort reads are fit softly (each with its rounding SD) over a library built for
newcomers: plain applicants who look like the existing field, pasters of published splits
and chatbot answers, and leaderboard trackers at levels 1-3 (level 1 best-responds to the
visible board, level 2 to the board plus level 1, ...). The fitted shares say how many
newcomers are trackers and how deep they go, which then sets the late-mover prior.
"""

from typing import NamedTuple

import numpy as np

from live import board_fit as bf
from sim import castles as c
from live import scrape_leaderboard as sl

ROUND_SD = 0.001 / np.sqrt(12)  # SD of one 3-dp rounding error
MODEL_SD = 0.05                 # floor for library misspecification, points per newcomer
SOURCES = ("plain", "pasted", "L1", "L2", "L3")
PRIOR_MASS = {"plain": 0.50, "pasted": 0.15, "L1": 0.15, "L2": 0.10, "L3": 0.10}
TRACKER_LAMS = (20.0, 60.0, 200.0)
MIN_COHORT = 20  # below this the read is too noisy to use (26 newcomers already gives +-0.1)


class CohortRead(NamedTuple):
    label: str
    alloc: np.ndarray
    mean_payoff: float  # anchor's average points against the newcomers
    sd: float
    k: int              # newcomers in the window
    removed: int        # entries that left the board in the window (biases the read)


class Cohort(NamedTuple):
    reads: list
    field: tuple        # (allocs, weights) fitted newcomer mix, or None
    shares: dict        # source -> fitted share of newcomers
    share_sd: dict      # source -> bootstrap SD
    castle4: dict       # share of newcomers sending exactly 46..50 on castle 4
    k: int


def cohort_reads(anchors, first, last):
    """Anchors' average payoff against unknown entries that arrived between two snapshots.

    `anchors` is a list of (label, board_id, alloc). Snapshots are the raw dicts. Anchors
    that are themselves newcomers (e.g. test entries) have known splits, so
    their contribution is subtracted exactly and they don't count toward the cohort.
    """
    ids0 = {e["id"] for e in first["g1"]}
    ids1 = {e["id"] for e in last["g1"]}
    new_ids = ids1 - ids0
    known_new = [alloc for _, board_id, alloc in anchors if board_id in new_ids]
    k, removed = len(new_ids) - len(known_new), len(ids0 - ids1)
    if k <= 0:
        return []
    n0, n1 = len(first["g1"]), len(last["g1"])
    s0 = {e["id"]: e["p"] for e in first["g1"]}
    s1 = {e["id"]: e["p"] for e in last["g1"]}
    sd = np.hypot(n0 - 1, n1 - 1) * ROUND_SD / k
    out = []
    known_new = np.array(known_new).reshape(-1, c.N_CASTLES)
    for label, board_id, alloc in anchors:
        if board_id in s0 and board_id in s1:
            total = (n1 - 1) * s1[board_id] - (n0 - 1) * s0[board_id]
            total -= bf.payoff_vs(alloc, known_new).sum() if len(known_new) else 0.0
            out.append(CohortRead(label, alloc, total / k, float(np.hypot(sd, MODEL_SD)), k, removed))
    return out


def newcomer_library(existing_allocs, existing_w, rng, n=3000):
    """Candidate newcomer splits, with each row tagged by the source that holds most of its mass."""
    seen = c.marginals(existing_allocs, existing_w)
    plain = existing_allocs[rng.choice(len(existing_allocs), n, p=existing_w)]
    chat = bf.chatbot_answers("game1")
    pasted = np.concatenate([np.array(list(bf.KNOWN_SPLITS.values())), chat]) if len(chat) \
        else np.array(list(bf.KNOWN_SPLITS.values()))
    parts = [(plain, PRIOR_MASS["plain"], "plain"), (pasted, PRIOR_MASS["pasted"], "pasted")]
    view, sharp_mass = seen, 0.3  # level k+1 assumes 30% of the field plays level k's best response
    for level in ("L1", "L2", "L3"):
        samples = np.concatenate([c.logit_sample(c.win_value(view), lam, n // 3, rng)
                                  for lam in TRACKER_LAMS])
        parts.append((samples, PRIOR_MASS[level], level))
        br, _ = c.best_response(c.win_value(view))
        view = (1 - sharp_mass) * seen + sharp_mass * c.marginals(br)
    return bf._pool(parts)


def soft_tilt(prior, allocs, reads, iters=50):
    """Minimize KL(w || prior) + sum_a (E_w[payoff_a] - read_a)^2 / (2 sd_a^2).

    Solved in its dual, w ~ prior * exp(theta . g): F(theta) = log Z - theta . m
    + 0.5 * sum sd^2 theta^2 is convex, with gradient E[g] - m + sd^2 theta and Hessian
    Cov(g) + diag(sd^2), so damped Newton converges in a handful of steps.
    """
    g = np.stack([bf.payoff_vs(r.alloc, allocs) for r in reads])  # (A, M)
    m = np.array([r.mean_payoff for r in reads])
    s2 = np.array([r.sd for r in reads]) ** 2
    logp = np.log(np.maximum(prior, 1e-300))

    def weights(theta):
        z = logp + theta @ g
        w = np.exp(z - z.max())
        return w / w.sum()

    def objective(theta):
        z = logp + theta @ g
        top = z.max()
        return top + np.log(np.exp(z - top).sum()) - theta @ m + 0.5 * s2 @ theta**2

    theta = np.zeros(len(reads))
    for _ in range(iters):
        w = weights(theta)
        mean = g @ w
        grad = mean - m + s2 * theta
        if np.max(np.abs(grad)) < 1e-6:
            break
        cov = (g * w) @ g.T - np.outer(mean, mean)
        step = np.linalg.solve(cov + np.diag(s2), grad)
        t, f0 = 1.0, objective(theta)
        while objective(theta - t * step) > f0 and t > 1e-4:
            t /= 2
        theta = theta - t * step
    return weights(theta)


def track(anchors, existing_allocs, existing_w, files=None, rng=None, n_boot=40):
    """Fit the newcomer cohort between the first and latest snapshots."""
    rng = np.random.default_rng(0) if rng is None else rng
    files = sl.snapshot_files() if files is None else files
    first, last = sl.read_snapshot(files[0]), sl.read_snapshot(files[-1])
    reads = cohort_reads(anchors, first, last)
    if not reads:
        return Cohort([], None, {}, {}, {}, 0)
    lib = newcomer_library(existing_allocs, existing_w, rng)

    def summarize(w):
        shares = {s: float(w[lib.source == s].sum()) for s in SOURCES}
        marg = c.marginals(lib.allocs, w)
        return shares, {x: float(marg[3, x]) for x in range(46, 51)}

    w = soft_tilt(lib.prior, lib.allocs, reads)
    shares, castle4 = summarize(w)
    boots = []
    for _ in range(n_boot):
        noisy = [r._replace(mean_payoff=r.mean_payoff + rng.normal(0, r.sd)) for r in reads]
        boots.append(summarize(soft_tilt(lib.prior, lib.allocs, noisy))[0])
    share_sd = {s: float(np.std([b[s] for b in boots])) for s in SOURCES}
    return Cohort(reads, (lib.allocs, w), shares, share_sd, castle4, reads[0].k)


def late_prior(cohort, base):
    """Late-mover settings informed by the observed cohort, or `base` if it is too small.

    Late entrants are assumed at least as strategic as the observed newcomers (the
    deadline crowd skews toward people who waited on purpose), so the tracker-share range
    starts at the observed share and extends 0.25 above it.
    """
    if cohort.k < MIN_COHORT:
        return base
    sharp = sum(cohort.shares[s] for s in ("L1", "L2", "L3"))
    levels = np.array([cohort.shares[s] for s in ("L1", "L2", "L3")]) / max(sharp, 1e-9)
    out = dict(base)
    out["sharp_share"] = (max(0.05, sharp), min(0.9, sharp + 0.25))
    out["level_mix"] = tuple(3 * levels + 0.5)  # Dirichlet centred on the observed mix
    return out
