"""Game 1 pick: anchored field ensemble x hidden late movers, ranked by mean score.

1. Ensemble. Several priors over who the existing applicants are (thinking depth, how
   many habit / sharp / equilibrium players), each tilted to the anchors and to "the best
   achievable score equals the board top". Weighted by leave-one-out anchor fit. The
   posted Boston leader split (5-7-35-47-6, live score via its posters' entries) is the
   anchor that rules out the stack-castle-5 family: no prior that predicts it can keep
   castle 5 stacked.
2. Worlds. For each member, sample the late applicants we can't see when we submit:
   ordinary procrastinators and copies of public splits (L0), sharp late movers who
   best-respond to the visible board (L1), and deeper ones who expect L1 (L2).
3. Selection. Candidates are ranked by mean score across worlds. P(#1) (score, 3 dp as
   the site rounds, at least every competitor's: the board's top cluster recomputed
   against the final field plus every late applicant) and P(top 10) are reported too.
"""

from typing import NamedTuple

import numpy as np

from live import board_fit as bf
from sim import castles as c

# Prior variants for the existing field: (tau, source-mass profile). Judgment calls that
# span shallow-to-deep crowds and habit-heavy to sharp-heavy mixes.
MASS_PROFILES = {
    "balanced": bf.SOURCE_MASS,
    "habit-heavy": {"chatbot": 0.10, "habit": 0.40, "lattice": 0.08, "focal": 0.07, "levelk": 0.20,
                    "equilibrium": 0.05, "sharp": 0.20},
    "sharp-heavy": {"chatbot": 0.10, "habit": 0.15, "lattice": 0.03, "focal": 0.07, "levelk": 0.25,
                    "equilibrium": 0.10, "sharp": 0.40},
    "equilibrium-heavy": {"chatbot": 0.10, "habit": 0.20, "lattice": 0.05, "focal": 0.05, "levelk": 0.20,
                          "equilibrium": 0.25, "sharp": 0.25},
}
TAUS = (1.5, 2.5, 3.5)

# Late applicants: everyone who enters after the last board we see (~10:06 PM ET Monday).
# Sharp late entrants are people running this same method: they track the leaderboard
# JSON, fit score anchors, and best-respond. They don't share our exact prior, so each
# world gives them their own estimate of the field (another ensemble member), and they
# split across thinking levels: L1 best-responds to the board, L2 to the board plus L1
# (they expect other trackers), L3 to the board plus L2.
LATE = {
    "count": (100, 600),          # uniform; Saturday ran ~22/hour, deadline evenings run hotter
    "sharp_share": (0.20, 0.60),  # user is certain many others track the board with anchors
    "level_mix": (1.0, 1.0, 1.0), # Dirichlet over L1/L2/L3: no strong view on depth
    "copy_share": 0.20,           # share of non-sharp late entrants pasting a public split
    "lams": (20.0, 60.0, 200.0),  # tracker precision; 200 is effectively an exact best response
}
LOO_SIGMA = 0.5  # points; soft, so no single prior dominates on a handful of anchors

class Member(NamedTuple):
    name: str
    allocs: np.ndarray
    weights: np.ndarray
    loo: np.ndarray       # leave-one-out anchor errors
    br: np.ndarray        # best response to this member's existing field
    competitors: np.ndarray  # stand-ins for the board's top entries


class Worlds(NamedTuple):
    member: np.ndarray       # (W,) index into the ensemble
    final_w: np.ndarray      # (W, 5, 101) win-value tables of each final field
    best_rival: np.ndarray   # (W,) best competitor score, rounded to 3 dp
    rivals: list             # per world, all competitor scores (for ranks)
    params: np.ndarray       # (W, 6) late count, sharp share, L1/L2/L3 shares of sharp, lam


def _loo(anchors, allocs, prior):
    errors = []
    for i, (alloc, score) in enumerate(anchors):
        w = bf.tilt_to_anchors(prior, allocs, anchors[:i] + anchors[i + 1:], sweeps=50)
        errors.append(c.expected_score(c.win_value(c.marginals(allocs, w)), alloc[None])[0] - score)
    return np.array(errors)


def _top_cluster_standins(wv, board_top_scores, rng):
    """Allocations whose score against the existing field matches each of the board's top scores.

    The board's leaders are near-best responses to the existing field; drawing sharp logit
    responses and matching them to the board's top scores by rank keeps their score levels
    honest while letting their exact splits vary from member to member.
    """
    pool = np.concatenate([c.logit_sample(wv, lam, 3000, rng) for lam in (20.0, 60.0, 200.0)])
    pool = np.unique(pool, axis=0)
    s = c.expected_score(wv, pool)
    order = np.argsort(-s)
    picks = []
    for target in np.sort(board_top_scores)[::-1]:
        j = order[np.argmin(np.abs(s[order] - target))]
        picks.append(pool[j])
    return np.array(picks)


def build_ensemble(anchors, board_scores, rng, taus=TAUS, profiles=MASS_PROFILES,
                   n_lib=3000, n_top=200):
    """One Member per (tau, mass profile); `anchors` is a list of (alloc, score)."""
    top = float(np.max(board_scores))
    top_scores = np.sort(board_scores)[-n_top:]
    members = []
    for tau in taus:
        for pname, mass in profiles.items():
            lib = bf.build_library(rng, n=n_lib, mass=mass, tau=tau)
            loo = _loo(anchors, lib.allocs, lib.prior)
            w = bf.tilt_to_board_top(lib.prior, lib.allocs, anchors, top, sweeps=50)
            # Sharp players respond to the real (tilted) field, not to the prior: rebuild once.
            lib = bf.build_library(rng, sharp_marg=c.marginals(lib.allocs, w), n=n_lib,
                                   mass=mass, tau=tau)
            w = bf.tilt_to_board_top(lib.prior, lib.allocs, anchors, top, sweeps=50)
            wv = c.win_value(c.marginals(lib.allocs, w))
            br, _ = c.best_response(wv)
            members.append(Member(f"tau={tau} {pname}", lib.allocs, w, loo, br,
                                  _top_cluster_standins(wv, top_scores, rng)))
    return members


def member_weights(members, sigma=LOO_SIGMA):
    """Weight each prior by how well it predicts anchors it wasn't fit to."""
    logw = np.array([-0.5 * np.sum(m.loo**2) / sigma**2 for m in members])
    w = np.exp(logw - logw.max())
    return w / w.sum()


def _late_entrants(member, belief, n_existing, count, sharp_share, level_mix, lam, rng,
                   public=None, cohort=None):
    """Allocations of the hidden late applicants for one world.

    `member` is the true existing field; `belief` is the field the sharp trackers think
    they see (their own anchor fit), which drives every level of their reasoning.
    `cohort` (allocs, weights), when given, is the fitted mix of newcomers seen so far
    (newcomers.track); the non-tracker late entrants are drawn from it instead of from
    the existing field, since late entrants look more like recent arrivals than like
    the applicants of a week ago.
    """
    n_sharp = int(round(sharp_share * count))
    n_levels = rng.multinomial(n_sharp, level_mix)
    n_plain = count - n_sharp
    if cohort is not None:
        allocs, w = cohort
        plain = allocs[rng.choice(len(allocs), n_plain, p=w)]
        n_copy = 0
    else:
        n_copy = int(round(LATE["copy_share"] * n_plain))
        plain = member.allocs[rng.choice(len(member.allocs), n_plain - n_copy, p=member.weights)]
    if n_copy == 0:
        copies = np.empty((0, c.N_CASTLES), dtype=np.int64)
    elif public is None:
        # Half paste a published split, half paste a chatbot answer (at real frequencies).
        known = np.array(list(bf.KNOWN_SPLITS.values()))
        chat = bf.chatbot_answers("game1")
        chat = chat if len(chat) else known
        n_chat = n_copy // 2
        copies = np.concatenate([known[rng.integers(0, len(known), n_copy - n_chat)],
                                 chat[rng.integers(0, len(chat), n_chat)]])
    else:
        public = np.asarray(public)
        copies = public[rng.integers(0, len(public), n_copy)] if n_copy else public[:0]

    seen = c.marginals(belief.allocs, belief.weights)
    out, view = [plain, copies], seen
    for n_k in n_levels:
        out.append(c.logit_sample(c.win_value(view), lam, n_k, rng))
        # The next level expects every sharp late entrant to play this level's best response.
        br_k, _ = c.best_response(c.win_value(view))
        view = (n_existing * seen + n_sharp * c.marginals(br_k)) / (n_existing + n_sharp)
    return np.concatenate(out)


def sample_worlds(members, probs, n_existing, n_worlds, rng, public=None, late=None,
                  cohort=None):
    late = LATE if late is None else late
    idx = rng.choice(len(members), n_worlds, p=probs)
    final_w, best, rivals, params = [], [], [], []
    for m_i in idx:
        m = members[m_i]
        belief = members[rng.choice(len(members), p=probs)]
        count = int(rng.integers(*late["count"]))
        sharp = rng.uniform(*late["sharp_share"])
        mix = rng.dirichlet(late["level_mix"])
        lam = float(rng.choice(late["lams"]))
        entrants = _late_entrants(m, belief, n_existing, count, sharp, mix, lam, rng, public,
                                  cohort)
        marg = (n_existing * c.marginals(m.allocs, m.weights)
                + len(entrants) * c.marginals(entrants)) / (n_existing + len(entrants))
        wv = c.win_value(marg)
        comp = np.concatenate([c.expected_score(wv, m.competitors),
                               c.expected_score(wv, entrants)])
        comp = np.round(comp, 3)
        final_w.append(wv)
        best.append(comp.max())
        rivals.append(np.sort(comp)[::-1][:200])
        params.append((count, sharp, *mix, lam))
    return Worlds(idx, np.array(final_w), np.array(best), rivals, np.array(params))


def neighbors(alloc):
    """Every allocation one soldier away (moved between two castles)."""
    out = []
    for i in range(c.N_CASTLES):
        for j in range(c.N_CASTLES):
            if i != j and alloc[i] > 0:
                b = alloc.copy()
                b[i] -= 1
                b[j] += 1
                out.append(b)
    return np.array(out)


def candidate_pool(members, worlds, per_field=100, rng=None):
    """Top allocations under a few representative final fields, every world's best response,
    and one-soldier neighbors of the strongest of those."""
    allocs = c.all_allocations()
    cands = [np.array([c.best_response(wv)[0] for wv in worlds.final_w])]
    for m_i in np.unique(worlds.member):
        rep = worlds.final_w[worlds.member == m_i].mean(axis=0)
        s = c.expected_score(rep, allocs)
        top = np.argpartition(-s, per_field)[:per_field]
        cands.append(allocs[top].astype(np.int64))
    pool = np.unique(np.concatenate(cands), axis=0)
    mean = np.mean([c.expected_score(wv, pool) for wv in worlds.final_w], axis=0)
    strongest = pool[np.argsort(-mean)[:60]]
    pool = np.unique(np.concatenate([pool] + [neighbors(a) for a in strongest]), axis=0)
    return pool


class Result(NamedTuple):
    allocs: np.ndarray
    p_first: np.ndarray
    p_top10: np.ndarray
    mean_score: np.ndarray
    mean_rank: np.ndarray
    bet_levels: np.ndarray  # (C, 3) average L1/L2/L3 mix in the worlds each candidate wins
    bet_sharp: np.ndarray   # average sharp share in the worlds each candidate wins


def evaluate(pool, worlds):
    scores = np.round(np.stack([c.expected_score(wv, pool) for wv in worlds.final_w]), 3)  # (W, C)
    wins = scores >= worlds.best_rival[:, None]
    ranks = np.stack([1 + (r[None, :] > s[:, None]).sum(axis=1)
                      for r, s in zip(worlds.rivals, scores)])
    n_wins = np.maximum(wins.sum(axis=0), 1)
    levels = (wins.T.astype(float) @ worlds.params[:, 2:5]) / n_wins[:, None]
    return Result(pool, wins.mean(axis=0), (ranks <= 10).mean(axis=0), scores.mean(axis=0),
                  ranks.mean(axis=0), levels, (wins * worlds.params[:, 1:2]).sum(axis=0) / n_wins)


def rank_candidates(res):
    """Order by mean score across worlds, then P(top 10).

    P(#1) was the first objective, but on 12 synthetic worlds with hidden late trackers
    (validate.py) it never produced a #1 and had a worse mean rank (205 vs 179) than
    ranking by mean score, and the model's own P(#1) was far too confident. Mean score is
    what the validation supports; P(#1) is still reported.
    """
    return np.lexsort((-res.p_top10, -res.mean_score))


def run(anchors, board_scores, rng, n_worlds=300, public=None, late=None, cohort=None,
        **ens_kw):
    """`public` lists splits late entrants may paste (default: the real published ones);
    `late` overrides LATE; `cohort` is the fitted newcomer mix (allocs, weights)."""
    members = build_ensemble(anchors, board_scores, rng, **ens_kw)
    probs = member_weights(members)
    worlds = sample_worlds(members, probs, len(board_scores), n_worlds, rng, public, late,
                           cohort)
    pool = candidate_pool(members, worlds)
    res = evaluate(pool, worlds)
    return members, probs, worlds, res
