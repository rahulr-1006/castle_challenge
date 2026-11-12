"""Build the worlds and the prior over them, run both games, and write results/simulation/.

    python -m sim.experiments           # full run
    python -m sim.experiments --quick   # small samples, a smoke test of the pipeline

One seeded generator drives every random draw, so a run is reproducible.
"""

import argparse
import itertools
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from sim import castles as c

RESULTS = Path(__file__).resolve().parents[1] / "results" / "simulation"
SEED = 2026

# ---------------------------------------------------------------------------
# Worlds: each is one guess at who the other applicants are
# ---------------------------------------------------------------------------

LEVEL0 = ("focal", "random")
# Inverse temperatures (Game 1 per castle point, Game 2 per payout point), set so a
# strategic player facing the round-number crowd gives up about 3% ("low") or
# 11-12% ("high") of the best achievable score. results/simulation/noise_calibration.csv
# records the actual figures.
NOISE = {"low": (8.0, 4.0), "high": (3.0, 1.0)}
MAX_LEVEL = 8  # Poisson tail beyond this is pooled; it holds 2% of players at tau = 4

# Priors over worlds. Axes are treated as independent; each weight carries its
# reason. "sharp" is the main one (results/simulation/); "baseline" is our first, more
# skeptical prior, kept for comparison (results/simulation/baseline/). Edit and rerun.
PRIORS = {}
PRIORS["sharp"] = {
    # Mean thinking depth of the Poisson cognitive hierarchy. tau = 2.5 puts about
    # half the thinkers at levels 2-3 and a quarter at 4+.
    "tau": {
        1.5: (0.10, "Floor: even a sharp pool has people who answer a 3-minute optional "
                    "question quickly."),
        2.0: (0.20, "Sharp applicants who stop at 'beat the people who beat the obvious "
                    "split'."),
        2.5: (0.35, "Core belief: Sequoia / OpenAI / Citadel draws a pool where most think "
                    "2-3 levels deep."),
        3.0: (0.25, "Priority registration rewards top scores, so many will think harder "
                    "than usual."),
        4.0: (0.10, "Tail: a pool where level 4+ is common."),
    },
    "eq": {
        0.0: (0.25, "Even in a sharp pool, few recall the Blotto equilibrium under time "
                    "pressure."),
        0.1: (0.45, "Quant applicants who know the theory randomize around proportional."),
        0.25: (0.30, "A pool selected for game theory has many equilibrium players."),
    },
    "meta": {
        0.0: (0.15, "Unlikely that nobody models the crowd in a pool this competitive."),
        0.1: (0.45, "A motivated minority models the field for priority registration."),
        0.25: (0.40, "Sharp pool: many will do some version of this analysis."),
    },
    "level0": {
        "focal": (0.80, "Whoever doesn't think still piles onto round numbers."),
        "random": (0.20, "Textbook uniform level 0, kept as a stress test."),
    },
    "noise": {
        "low": (0.60, "Sharp players mostly get their arithmetic right."),
        "high": (0.40, "Three minutes still makes some of them sloppy."),
    },
}
PRIORS["baseline"] = {
    # Mean thinking depth of the Poisson cognitive hierarchy.
    "tau": {
        0.5: (0.15, "Optional 3-minute question: many applicants answer on autopilot."),
        1.0: (0.30, "Most who think at all stop after one step: beat the obvious split."),
        1.5: (0.30, "Typical lab estimate (Camerer, Ho and Chong 2004); a technical pool "
                    "should reach it."),
        2.0: (0.15, "Sequoia / OpenAI / Citadel applicants skew quantitative and competitive."),
        3.0: (0.10, "Tail: priority registration draws a motivated subset that thinks "
                    "several steps ahead."),
    },
    # Share of all players who know Blotto theory and play its mixed equilibrium.
    "eq": {
        0.0: (0.40, "Few people derive or recall the Blotto equilibrium in 3 minutes."),
        0.1: (0.45, "Some quant applicants know it (randomize around proportional) and use "
                    "it as a safe default."),
        0.25: (0.15, "A pool selected for game theory could hold many equilibrium players."),
    },
    # Share of all players who best-respond to the actual crowd, as this repo does.
    "meta": {
        0.0: (0.35, "Modeling the crowd takes far more than 3 minutes; most won't."),
        0.1: (0.45, "Top scores earn priority registration, so a motivated minority will "
                    "model the field."),
        0.25: (0.20, "Stress case: a pool full of people doing exactly this analysis."),
    },
    # What non-thinking (level 0) players look like.
    "level0": {
        "focal": (0.80, "People in Blotto-style contests pile onto round numbers and simple "
                        "shapes."),
        "random": (0.20, "Textbook uniform level 0, kept as a stress test of that assumption."),
    },
    # How sloppy strategic players are.
    "noise": {
        "low": (0.40, "Players who take the time to check their arithmetic."),
        "high": (0.60, "Three minutes and no computer: most strategic players will be "
                       "roughly, not exactly, right."),
    },
}


class World(NamedTuple):
    tau: float
    eq: float
    meta: float
    level0: str
    noise: str
    weight: float

    def describe(self):
        return (f"tau={self.tau:g} eq={self.eq:g} meta={self.meta:g} "
                f"L0={self.level0} noise={self.noise}")


PRIOR = PRIORS["sharp"]
TAUS = tuple(sorted(PRIOR["tau"]))


def use_prior(name):
    """Point the module at a named prior and its results folder."""
    global PRIOR, TAUS, RESULTS
    PRIOR = PRIORS[name]
    TAUS = tuple(sorted(PRIOR["tau"]))
    RESULTS = Path(__file__).resolve().parents[1] / "results" / "simulation" / ("" if name == "sharp" else name)


def build_worlds(prior=None):
    prior = PRIOR if prior is None else prior
    for axis, weights in prior.items():
        total = sum(w for w, _ in weights.values())
        assert abs(total - 1) < 1e-9, f"prior weights for {axis} sum to {total}"
    axes = [prior[a].items() for a in ("tau", "eq", "meta", "level0", "noise")]
    return [World(*(k for k, _ in combo), float(np.prod([w for _, (w, _) in combo])))
            for combo in itertools.product(*axes)]


def crowd_weights(world):
    """Shares of each CH level and of equilibrium players among the non-exploiters."""
    ch = (1 - world.eq - world.meta) * c.poisson_levels(world.tau, MAX_LEVEL)
    return ch / (1 - world.meta), world.eq / (1 - world.meta)


class Sizes(NamedTuple):
    fp_rounds: int       # fictitious play rounds for the Game 1 equilibrium
    level0_game1: int    # samples behind each level-0 flavor's Game 1 marginals
    level0_game2: int    # allocations kept as each Game 2 level-0 field
    br_groups: int       # groups per Game 2 best-response search
    eq_rounds: int       # fictitious play rounds for the Game 2 equilibrium
    eq_steps: int        # replicator steps (comparison only)
    search_groups: int   # prior-mixture groups for the Game 2 candidate search
    anneal_steps: int
    eval_groups: int     # groups per candidate per world in the final comparison


FULL = Sizes(100_000, 400_000, 60_000, 4000, 1500, 300, 40_000, 3000, 20_000)
QUICK = Sizes(5_000, 50_000, 10_000, 800, 60, 30, 4000, 150, 1500)

_START = time.time()


def log(msg):
    print(f"[{time.time() - _START:6.1f}s] {msg}", flush=True)


def fmt(alloc):
    return "-".join(str(int(x)) for x in alloc)


class Candidates:
    """Named allocations; a repeated allocation keeps its first name."""

    def __init__(self):
        self.names, self.allocs = [], []

    def add(self, name, alloc):
        alloc = c.validate(alloc)
        if not any((alloc == a).all() for a in self.allocs):
            self.names.append(name)
            self.allocs.append(alloc)

    def array(self):
        return np.array(self.allocs)


def summarize(cands, scores, best, worlds):
    """Prior score, worst world and regret for each candidate (scores: candidates x worlds)."""
    p = np.array([w.weight for w in worlds])
    regret = best[None, :] - scores
    worst = scores.argmin(axis=1)
    return pd.DataFrame({
        "candidate": cands.names,
        "allocation": [fmt(a) for a in cands.allocs],
        "prior_score": scores @ p,
        "worst_score": scores.min(axis=1),
        "worst_world": [worlds[i].describe() for i in worst],
        "prior_regret": regret @ p,
        "max_regret": regret.max(axis=1),
    })


def best_by_tau(table, value_cols):
    """For each tau row, the best column; and where the leader changes (crossovers)."""
    leaders = table[value_cols].idxmax(axis=1)
    crossovers = []
    for (t0, a), (t1, b) in itertools.pairwise(zip(table["tau"], leaders)):
        if a != b:
            # Linear interpolation of the tau where b's advantage over a turns positive.
            d0 = table.loc[table["tau"] == t0, b].item() - table.loc[table["tau"] == t0, a].item()
            d1 = table.loc[table["tau"] == t1, b].item() - table.loc[table["tau"] == t1, a].item()
            crossovers.append({"from": a, "to": b, "tau": t0 + (t1 - t0) * d0 / (d0 - d1)})
    return leaders, pd.DataFrame(crossovers, columns=["from", "to", "tau"])


# ---------------------------------------------------------------------------
# Game 1: everything is exact (separable scores, DP best responses)
# ---------------------------------------------------------------------------

GAME1_SEEDS = {
    "equal split": [20, 20, 20, 20, 20],
    "proportional": [7, 13, 20, 27, 33],
    "concede 1": [0, 14, 21, 29, 36],
    "concede 1-2": [0, 0, 25, 33, 42],
    "concede 5": [10, 20, 30, 40, 0],
    "concede 5, focal+1": [7, 21, 31, 41, 0],
    "focal+1 (1-11-21-31-36)": [1, 11, 21, 31, 36],
    "focal+1 (0-16-21-26-37)": [0, 16, 21, 26, 37],
    "focal+1 (6-11-21-26-36)": [6, 11, 21, 26, 36],
}


def game1_field(world, levels, eq_marg):
    """Per-castle marginals of a world's whole field in Game 1."""
    ch_w, eq_w = crowd_weights(world)
    crowd = np.tensordot(ch_w, levels, axes=1) + eq_w * eq_marg
    meta = c.noisy_response(c.win_value(crowd), NOISE[world.noise][0])
    return (1 - world.meta) * crowd + world.meta * meta


def exhaustive_game1(w, p, best):
    """Prior score, worst-world score and max regret of all 4.6M allocations."""
    allocs = c.all_allocations()
    expected = np.zeros(len(allocs))
    worst = np.full(len(allocs), np.inf)
    max_regret = np.zeros(len(allocs))
    for w_i, p_i, b_i in zip(w, p, best):
        s = c.expected_score(w_i, allocs)
        expected += p_i * s
        np.minimum(worst, s, out=worst)
        np.maximum(max_regret, b_i - s, out=max_regret)
    return expected, worst, max_regret


def game1_tau_sweep(level0, eq_marg, allocs, taus):
    """Expected score of each allocation at each tau, other axes averaged by the prior."""
    rows = []
    others = list(itertools.product(PRIOR["eq"].items(), PRIOR["meta"].items(),
                                    PRIOR["level0"].items(), PRIOR["noise"].items()))
    for tau in taus:
        levels = {(f, n): c.cognitive_hierarchy(level0[f], tau, NOISE[n][0], MAX_LEVEL)
                  for f in LEVEL0 for n in NOISE}
        w_tau = 0
        for (eq, (pe, _)), (meta, (pm, _)), (f, (pf, _)), (n, (pn, _)) in others:
            world = World(tau, eq, meta, f, n, pe * pm * pf * pn)
            w_tau = w_tau + world.weight * c.win_value(game1_field(world, levels[f, n], eq_marg))
        opt, opt_value = c.best_response(w_tau)
        rows.append((tau, c.expected_score(w_tau, allocs), opt, opt_value))
    return rows


def run_game1(worlds, rng, sizes):
    log("Game 1: fictitious play for the equilibrium")
    fp = c.fictitious_play(sizes.fp_rounds)
    rounds = np.unique(np.geomspace(1, sizes.fp_rounds, 200).astype(int))
    pd.DataFrame({"round": rounds, "exploitability": fp.exploitability[rounds - 1]}).to_csv(
        RESULTS / "fp_convergence.csv", index=False)
    pd.DataFrame(fp.marginals.T, columns=[f"castle_{i}" for i in c.VALUES]).rename_axis(
        "soldiers").to_csv(RESULTS / "game1_equilibrium_marginals.csv")
    log(f"  exploitability after {sizes.fp_rounds} rounds: {fp.exploitability[-1]:.4f} points")

    level0 = {f: c.marginals(c.level0_sample(f, sizes.level0_game1, rng)) for f in LEVEL0}
    ch = {(f, n, t): c.cognitive_hierarchy(level0[f], t, NOISE[n][0], MAX_LEVEL)
          for f, n, t in itertools.product(LEVEL0, NOISE, TAUS)}
    fields = np.array([game1_field(wd, ch[wd.level0, wd.noise, wd.tau], fp.marginals)
                       for wd in worlds])
    w = np.array([c.win_value(f) for f in fields])
    p = np.array([wd.weight for wd in worlds])
    best = [c.best_response(x) for x in w]
    best_value = np.array([v for _, v in best])
    w_prior = np.tensordot(p, w, axes=1)

    log("Game 1: exhaustive search over all 4,598,126 allocations in every world")
    expected, worst, max_regret = exhaustive_game1(w, p, best_value)
    allocs = c.all_allocations()
    bayes_dp, bayes_value = c.best_response(w_prior)
    # The prior score is separable too, so the DP must agree with exhaustive search.
    assert np.isclose(bayes_value, expected.max())

    cands = Candidates()
    cands.add("Bayes: best prior score", bayes_dp)
    cands.add("minimax regret", allocs[max_regret.argmin()])
    cands.add("maximin: best worst world", allocs[worst.argmax()])
    for name, a in GAME1_SEEDS.items():
        cands.add(name, a)
    cands.add("equilibrium mean (rounded)", c.mean_allocation(fp.marginals))
    for tau in (1.0, 1.5, 2.0):
        exact = c.cognitive_hierarchy(level0["focal"], tau, np.inf, 3)
        for k in (1, 2, 3):
            cands.add(f"CH level {k} best response (tau={tau:g})", exact[k].argmax(axis=1))

    # Decision principle test: how far above the field should we aim? CH levels
    # best-respond to everything below them; the level-k chain only to the level
    # just below, so it can overshoot.
    level_rows = []
    exact = c.cognitive_hierarchy(level0["focal"], 1.5, np.inf, 6)
    chain = level0["focal"]
    for k in range(1, 7):
        chain_alloc = c.best_response(c.win_value(chain))[0]
        chain = c.marginals(chain_alloc)
        for kind, a in (("CH (tau=1.5)", exact[k].argmax(axis=1)), ("level-k chain", chain_alloc)):
            scores = np.array([c.expected_score(x, a) for x in w])
            level_rows.append({"model": kind, "level": k, "allocation": fmt(a),
                               "prior_score": scores @ p, "prior_regret": (best_value - scores) @ p})
            cands.add(f"{kind} level {k}", a)
    levels_table = pd.DataFrame(level_rows)
    levels_table.to_csv(RESULTS / "game1_levels.csv", index=False)

    scores = np.array([c.expected_score(x, cands.array()) for x in w]).T
    table = summarize(cands, scores, best_value, worlds).sort_values("prior_score", ascending=False)
    table.to_csv(RESULTS / "game1_candidates.csv", index=False)
    pd.DataFrame(scores, index=cands.names, columns=[wd.describe() for wd in worlds]).to_csv(
        RESULTS / "game1_scores_by_world.csv")

    log("Game 1: tau sweep for crossovers")
    taus = np.round(np.arange(0.25, max(TAUS) + 0.51, 0.125), 3)
    sweep = game1_tau_sweep(level0, fp.marginals, cands.array(), taus)
    crossover = pd.DataFrame([dict(zip(cands.names, s)) | {"tau": t} for t, s, _, _ in sweep])
    leaders, switches = best_by_tau(crossover, cands.names)
    crossover["best_candidate"] = leaders
    crossover["optimum_at_tau"] = [fmt(o) for _, _, o, _ in sweep]
    crossover["optimum_score"] = [v for _, _, _, v in sweep]
    crossover.to_csv(RESULTS / "game1_crossover.csv", index=False)
    switches.to_csv(RESULTS / "game1_crossover_points.csv", index=False)

    # Fact 4: the value of one more soldier spikes just above crowded numbers.
    marg_prior = np.tensordot(p, fields, axes=1)
    step_value = np.diff(w_prior, axis=1)
    focal = pd.DataFrame([
        {"castle": int(v), "from": s, "to": s + 1, "field_share_at_s": marg_prior[i, s],
         "gain_points": step_value[i, s]}
        for i, v in enumerate(c.VALUES) for s in range(c.TOTAL_SOLDIERS)])
    focal.to_csv(RESULTS / "game1_focal_steps.csv", index=False)

    worlds_table = pd.DataFrame([wd._asdict() | {"game1_best": fmt(a), "game1_best_score": v}
                                 for wd, (a, v) in zip(worlds, best)])
    return {"fp": fp, "table": table, "levels": levels_table, "crossover": crossover,
            "switches": switches, "focal": focal, "cands": cands, "scores": scores,
            "worlds_table": worlds_table, "marg_prior": marg_prior, "w_prior": w_prior,
            "level0": level0, "ch": ch, "best_value": best_value, "fields": fields}


# ---------------------------------------------------------------------------
# Game 2: simulated, with the same groups for every candidate within a world
# ---------------------------------------------------------------------------

def top_plans(response, n):
    return response.plans[np.argsort(-response.values)[:n]]


def game2_levels(level0_field, lam, rng, sizes):
    """Game 2 CH fields for every tau: {tau: [level 0, level 1, ...]} plus all responses."""
    responses = [c.game2_response(level0_field, rng, sizes.br_groups)]
    level1 = c.logit_field(responses[0], lam)  # level 1 does not depend on tau
    by_tau = {}
    for tau in TAUS:
        shares = c.poisson_levels(tau, MAX_LEVEL)
        levels = [level0_field, level1]
        for k in range(2, MAX_LEVEL + 1):
            r = c.game2_response(c.mix_fields(zip(shares[:k], levels)), rng, sizes.br_groups)
            responses.append(r)
            levels.append(c.logit_field(r, lam))
        by_tau[tau] = levels
    return by_tau, responses


def game2_equilibrium(pool, rng, sizes):
    """Approximate Game 2 equilibrium by fictitious play; replicator dynamics on a
    fixed plan pool is run alongside as the comparison that motivated it."""
    marks = sorted({int(x) for x in np.geomspace(10, sizes.eq_rounds, 8)})
    plays, gaps = c.fictitious_play_game2(sizes.eq_rounds, rng, sizes.br_groups // 2, marks)
    trace = [{"method": "fictitious play", "rounds": t, "exploitability": g} for t, g in gaps.items()]
    rep = c.replicator(pool, rng, steps=sizes.eq_steps, n_groups=sizes.br_groups // 2)
    br = c.game2_response(rep, rng, 3 * sizes.br_groups // 2)
    trace.append({"method": "replicator on plan pool", "rounds": sizes.eq_steps,
                  "exploitability": br.values.max() - 3.75})
    return c.field_from_samples(plays), pd.DataFrame(trace)


def concentrate(alloc, drop):
    """Higher-variance version: move the soldiers on castles in `drop` onto the rest, by value."""
    a = np.array(alloc)
    keep = ~np.isin(c.VALUES, drop)
    moved = a[~keep].sum()
    a[~keep] = 0
    a[keep] += c.round_to_total(c.VALUES[keep] / c.VALUES[keep].sum(), 1, moved)[0]
    return a


def run_game2(worlds, rng, sizes, g1):
    p = np.array([wd.weight for wd in worlds])
    log("Game 2: cognitive hierarchy best responses")
    level0 = {f: c.field_from_samples(c.level0_sample(f, sizes.level0_game2, rng)) for f in LEVEL0}
    ch, responses = {}, []
    for f, n in itertools.product(LEVEL0, NOISE):
        by_tau, rs = game2_levels(level0[f], NOISE[n][1], rng, sizes)
        responses += rs
        for tau, levels in by_tau.items():
            ch[f, n, tau] = levels
    log(f"  {len(responses)} best responses")

    log("Game 2: approximate equilibrium (fictitious play)")
    pool = np.unique(np.vstack([top_plans(r, 3) for r in responses] + [g1["cands"].array()]), axis=0)
    eq_field, eq_trace = game2_equilibrium(pool, rng, sizes)
    eq_trace.to_csv(RESULTS / "game2_equilibrium_trace.csv", index=False)
    pd.DataFrame(eq_field.allocs, columns=[f"castle_{i}" for i in c.VALUES]).to_csv(
        RESULTS / "game2_equilibrium_plays.csv", index=False)
    log("  exploitability: " + ", ".join(
        f"{r.method} {r.rounds}: {r.exploitability:.3f}" for r in eq_trace.itertuples()))

    log("Game 2: meta exploiters and world fields")
    fields, exploiters = [], {}
    for wd in worlds:
        ch_w, eq_w = crowd_weights(wd)
        crowd = c.mix_fields(list(zip(ch_w, ch[wd.level0, wd.noise, wd.tau])) + [(eq_w, eq_field)])
        key = (wd.level0, wd.noise, wd.tau, wd.eq, wd.meta)
        if wd.meta > 0 and key not in exploiters:
            exploiters[key] = c.logit_field(c.game2_response(crowd, rng, sizes.br_groups), NOISE[wd.noise][1])
        fields.append(c.mix_fields([(1 - wd.meta, crowd), (wd.meta, exploiters.get(key))])
                      if wd.meta > 0 else crowd)

    log("Game 2: candidates and search")
    cands = Candidates()
    g1_table = g1["table"].set_index("candidate")
    for name in g1_table.index[:3]:
        cands.add(f"Game 1: {name}", g1["cands"].allocs[g1["cands"].names.index(name)])
    for name in ("equal split", "proportional"):
        cands.add(name, GAME1_SEEDS[name])
    g1_best = cands.allocs[0]
    cands.add("Game 1 best, castle 1 dropped", concentrate(g1_best, [1]))
    cands.add("Game 1 best, castles 1-2 dropped", concentrate(g1_best, [1, 2]))
    for s, split in {(3, 5): (40, 60), (4, 5): (50, 50), (1, 2, 5): (20, 30, 50),
                     (1, 3, 4): (15, 35, 50), (2, 3, 4): (20, 35, 45)}.items():
        a = np.zeros(5, dtype=int)
        a[np.array(s) - 1] = split
        cands.add(f"sweep {set(s)}: round split", a)
    marg_prior = np.tensordot(p, [c.field_marginals(f) for f in fields], axes=1)
    for s in c.CASTLE_SETS:
        cands.add(f"sweep {set(s)}: DP vs prior field", c.sweep_plan(marg_prior, s))
    cands.add("expected points vs 3 (DP)",
              c.best_response(c.VALUES[:, None] * c.castle_share_4way(marg_prior))[0])

    counts = rng.multinomial(sizes.search_groups, p)
    opp = c.face(np.concatenate([c.sample_opponents(f, k, rng) for f, k in zip(fields, counts) if k]))
    est = c.payouts(cands.array(), opp).mean(axis=1)
    order = np.argsort(-est)
    for i in order[:5]:
        cands.add(f"local search from {cands.names[i]}", c.local_search(cands.allocs[i], opp)[0])
    for i in order[:2]:
        cands.add(f"annealed from {cands.names[i]}", c.anneal(cands.allocs[i], opp, sizes.anneal_steps, rng)[0])

    log(f"Game 2: final comparison, {len(cands.names)} candidates x {len(worlds)} worlds "
        f"x {sizes.eval_groups} groups")
    allocs = cands.array()
    n_c, n_w = len(allocs), len(worlds)
    mean, first = np.empty((n_c, n_w)), np.empty((n_c, n_w))
    cov = np.empty((n_w, n_c, n_c))
    best_known, world_best = np.empty(n_w), []
    for i, field in enumerate(fields):
        br = c.game2_response(field, rng, sizes.br_groups)
        br_alloc = br.plans[br.values.argmax()]
        groups = c.face(c.sample_opponents(field, sizes.eval_groups, rng))
        pay = c.payouts(np.vstack([allocs, br_alloc]), groups)
        mean[:, i] = pay[:-1].mean(axis=1)
        first[:, i] = (pay[:-1] == 9).mean(axis=1)
        cov[i] = np.cov(pay[:-1]) / sizes.eval_groups
        best_known[i] = pay.mean(axis=1).max()
        world_best.append((br_alloc, pay[-1].mean()))

    table = summarize(cands, mean, best_known, worlds)
    se = np.sqrt(np.einsum("w,wcc->c", p**2, cov))
    table.insert(3, "ci95", 1.96 * se)
    table.insert(4, "p_outright_first", first @ p)
    lead = int(np.argmax(mean @ p))
    diff_var = np.einsum("w,wc->c", p**2, cov[:, np.arange(n_c), np.arange(n_c)]
                         + cov[:, lead, lead][:, None] - 2 * cov[:, lead, :])
    table.insert(5, "gap_to_leader", mean[lead] @ p - mean @ p)
    table.insert(6, "gap_ci95", 1.96 * np.sqrt(diff_var))
    table = table.sort_values("prior_score", ascending=False)
    table.to_csv(RESULTS / "game2_candidates.csv", index=False)
    pd.DataFrame(mean, index=cands.names, columns=[wd.describe() for wd in worlds]).to_csv(
        RESULTS / "game2_scores_by_world.csv")

    crossover = pd.DataFrame([
        dict(zip(cands.names, mean[:, m] @ (p[m] / p[m].sum()))) | {"tau": tau}
        for tau in TAUS for m in [np.array([wd.tau == tau for wd in worlds])]])
    leaders, switches = best_by_tau(crossover, cands.names)
    crossover["best_candidate"] = leaders
    crossover.to_csv(RESULTS / "game2_crossover.csv", index=False)
    switches.to_csv(RESULTS / "game2_crossover_points.csv", index=False)

    worlds_table = g1["worlds_table"].assign(
        game2_best=[fmt(a) for a, _ in world_best], game2_best_payout=[v for _, v in world_best])
    worlds_table.to_csv(RESULTS / "worlds.csv", index=False)
    return {"table": table, "crossover": crossover, "switches": switches, "eq_trace": eq_trace,
            "cands": cands, "mean": mean, "fields": fields}


def noise_calibration(rng):
    """What each noise level costs a strategic player facing the round-number crowd."""
    rows = []
    level0 = c.level0_sample("focal", 200_000, rng)
    w = c.win_value(c.marginals(level0))
    _, best = c.best_response(w)
    r = c.game2_response(c.field_from_samples(level0[:60_000]), rng, 8000)
    for noise, (lam1, lam2) in NOISE.items():
        g1_loss = best - (c.logit_marginals(w, lam1) * w).sum()
        g2 = c.logit_field(r, lam2)
        g2_loss = r.values.max() - g2.probs @ r.values
        rows.append({"noise": noise, "game1_lambda": lam1, "game1_loss_points": g1_loss,
                     "game1_loss_share": g1_loss / best, "game2_lambda": lam2,
                     "game2_loss_payout": g2_loss, "game2_loss_share": g2_loss / r.values.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Stress test: one more layer of thinking
# ---------------------------------------------------------------------------

COUNTER_SHARES = (0.1, 0.25)


def run_counter_stress(worlds, rng, sizes, g1, g2):
    """Replace a share of each world with counter-players who best-respond to the whole
    field, crowd-modelers included (the layer nobody else in the model anticipates),
    then re-solve: does each game's pick survive?"""
    p = np.array([wd.weight for wd in worlds])
    rows = []
    g1_top = g1["table"].head(4)
    for share in COUNTER_SHARES:
        log(f"Counter-player stress test, share {share:g}: Game 1")
        w = np.array([
            c.win_value((1 - share) * f + share * c.noisy_response(c.win_value(f), NOISE[wd.noise][0]))
            for wd, f in zip(worlds, g1["fields"])])
        best = np.array([c.best_response(x)[1] for x in w])
        expected, _, max_regret = exhaustive_game1(w, p, best)
        bayes = c.all_allocations()[expected.argmax()]
        allocs = [("Bayes under this stress", bayes)] + [
            (name, np.array([int(x) for x in a.split("-")]))
            for name, a in zip(g1_top["candidate"], g1_top["allocation"])]
        for name, a in allocs:
            scores = np.array([c.expected_score(x, a) for x in w])
            rows.append({"game": 1, "counter_share": share, "candidate": name, "allocation": fmt(a),
                         "prior_score": scores @ p, "prior_regret": (best - scores) @ p,
                         "max_regret": (best - scores).max()})

        log(f"Counter-player stress test, share {share:g}: Game 2")
        g2_top = g2["table"].head(5)
        cands = np.array([[int(x) for x in a.split("-")] for a in g2_top["allocation"]])
        mean = np.empty((len(cands), len(worlds)))
        for i, (wd, field) in enumerate(zip(worlds, g2["fields"])):
            counter = c.logit_field(c.game2_response(field, rng, sizes.br_groups), NOISE[wd.noise][1])
            stressed = c.mix_fields([(1 - share, field), (share, counter)])
            mean[:, i] = c.payouts(cands, c.face(c.sample_opponents(stressed, sizes.eval_groups, rng))).mean(axis=1)
        for name, a, m in zip(g2_top["candidate"], g2_top["allocation"], mean):
            rows.append({"game": 2, "counter_share": share, "candidate": name, "allocation": a,
                         "prior_score": m @ p, "prior_regret": np.nan, "max_regret": np.nan})
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "counter_stress.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# Plots (read back from the CSVs, so --plots-only can redraw them)
# ---------------------------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three categorical slots, all-pairs safe
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _crossover_plot(ax, table, names, title, ylabel, optimum=None):
    for color, name in zip(SERIES, names):
        ax.plot(table["tau"], table[name], color=color, linewidth=2, marker="o", markersize=4,
                label=name)
    if optimum is not None:
        ax.plot(table["tau"], optimum, color=MUTED, linewidth=1.2, linestyle="--",
                label="best possible at each tau")
    _style(ax, title, "tau (mean thinking depth of the field)", ylabel)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)


def make_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    fp = pd.read_csv(RESULTS / "fp_convergence.csv")
    a1.loglog(fp["round"], fp["exploitability"], color=SERIES[0], linewidth=2)
    _style(a1, "Game 1: fictitious play converges", "round", "best-response gain (points)")
    eq = pd.read_csv(RESULTS / "game1_equilibrium_marginals.csv")
    for color, castle in zip(SERIES, (3, 4, 5)):
        a2.plot(eq["soldiers"], eq[f"castle_{castle}"].cumsum(), color=color, linewidth=2,
                label=f"castle {castle}")
        top = 200 * castle / 15
        a2.plot([0, top, 100], [0, 1, 1], color=MUTED, linewidth=1, linestyle="--")
    a2.plot([], [], color=MUTED, linewidth=1, linestyle="--", label="Gross-Wagner (continuous)")
    _style(a2, "Game 1 equilibrium vs theory", "soldiers sent", "cumulative probability")
    a2.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig(RESULTS / "equilibrium.png")
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    g1 = pd.read_csv(RESULTS / "game1_candidates.csv")
    cross1 = pd.read_csv(RESULTS / "game1_crossover.csv")
    _crossover_plot(a1, cross1, g1["candidate"].head(3).tolist(), "Game 1: score vs field depth",
                    "expected score (0-15)", cross1["optimum_score"])
    g2 = pd.read_csv(RESULTS / "game2_candidates.csv")
    cross2 = pd.read_csv(RESULTS / "game2_crossover.csv")
    _crossover_plot(a2, cross2, g2["candidate"].head(3).tolist(), "Game 2: payout vs field depth",
                    "expected payout (0-9)")
    fig.tight_layout()
    fig.savefig(RESULTS / "crossover.png")
    plt.close(fig)

    focal = pd.read_csv(RESULTS / "game1_focal_steps.csv")
    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8), sharey=True)
    for ax, (castle, rows) in zip(axes, focal.groupby("castle")):
        ax.bar(rows["to"], rows["gain_points"], width=0.9, color=SERIES[0])
        _style(ax, f"castle {castle}", "soldiers (s to s+1)", "points gained" if castle == 1 else "")
    fig.suptitle("Game 1: what one more soldier is worth under the prior (spikes sit just above "
                 "crowded numbers)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "focal_points.png")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small samples for a fast smoke test")
    parser.add_argument("--plots-only", action="store_true", help="redraw plots from results/simulation/")
    parser.add_argument("--prior", choices=sorted(PRIORS), default="sharp",
                        help="which prior to use (default: sharp)")
    args = parser.parse_args()
    use_prior(args.prior)
    if args.plots_only:
        make_plots()
        return
    sizes = QUICK if args.quick else FULL
    RESULTS.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    worlds = build_worlds()
    pd.DataFrame([
        {"axis": axis, "value": value, "weight": weight, "reason": reason}
        for axis, options in PRIOR.items() for value, (weight, reason) in options.items()
    ]).to_csv(RESULTS / "prior.csv", index=False)
    noise_calibration(rng).to_csv(RESULTS / "noise_calibration.csv", index=False)

    g1 = run_game1(worlds, rng, sizes)
    g2 = run_game2(worlds, rng, sizes, g1)
    stress = run_counter_stress(worlds, rng, sizes, g1, g2)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 45)
    cols = ["candidate", "allocation", "prior_score", "worst_score", "prior_regret", "max_regret"]
    print("\nGame 1 (exact)\n", g1["table"][cols].head(12).to_string(index=False))
    print("\nGame 1 crossovers\n", g1["switches"].to_string(index=False))
    print("\nGame 2 (simulated)\n", g2["table"][cols[:3] + ["ci95", "gap_to_leader", "gap_ci95",
                                                            "p_outright_first", "worst_score",
                                                            "max_regret"]].head(12).to_string(index=False))
    print("\nGame 2 crossovers\n", g2["switches"].to_string(index=False))
    print("\nCounter-player stress test\n", stress.round(3).to_string(index=False))
    make_plots()
    log("done")


if __name__ == "__main__":
    main()
