"""Infer other entries' Game 1 splits from their board scores, with calibrated confidence.

An entry's Game 1 score is S(a) = sum_c v_c G_c(a_c), and G comes from the field. Every
ensemble member m (robust.build_ensemble) is a fitted field, so it predicts S_m(a) for all
4.6M splits and says how common each split is, w_m(a). For an entry with board score s:

    P(a | s)  ∝  sum_m  pi_m * prior_m(a) * N(s; S_m(a), sigma^2)

- pi_m is the member's leave-one-out weight (robust.member_weights).
- prior_m(a) = (1 - eps) * w_m(a) + eps / 4.6M. The floor lets a split the library never
  sampled win when its score fits better than anything the library has.
- sigma is the model's score error. Spread between members already covers part of it.

What limits this is precision, not information. With the true field, the top 10 scores on
a synthetic board each belong to exactly one of the 4.6M legal splits, and ranks 11-50 to
2-3. But using that needs predicted scores good to about 0.0005, and the fitted field's
held-out error on the real board is 0.006-0.06 near the top and 0.1-0.6 below it. So the
posterior can't name exact splits; the question is which coarser facts it gets right.

(eps, sigma) are chosen per rank band on synthetic boards with a known field, then tested on
the real anchors, each held out in turn. They are never tuned on the anchors they're
tested on.

    .venv/bin/python -m live.infer_splits --validate --workers 4   # -> results/live/
    .venv/bin/python -m live.infer_splits --infer                  # -> data/inferred/
"""

import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from live import board_fit as bf
from live import robust as rb
from live import scrape_leaderboard as sl
from live import validate as vd
from live.anchors import fmt, load_anchors
from sim import castles as c

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "live"
PARAMS = RESULTS / "inference_params.csv"
INFERRED = ROOT / "data" / "inferred"  # gitignored: rank -> split is one lookup from a name

EPS_GRID = (1e-3, 1e-2, 1e-1)
SIGMA_GRID = (0.005, 0.02, 0.06, 0.2)  # points; spans the real held-out errors
WINDOW = 4.0               # half-width of the score window, in standard deviations
ROUND_VAR = 0.001**2 / 12  # the site rounds scores to 3 dp
BANDS = ((1, 10), (11, 50), (51, 200), (201, 1000), (1001, 3000), (3001, 10**6))
PER_BAND = 12
N_PROBES = 10  # our real test entries: ~10 one-soldier steps climbing from a posted split
CONCEDE5 = 15  # "concedes castle 5" means at most this many soldiers there
SEED = 2026

ALL = c.all_allocations()
N_ALL = len(ALL)
# all_allocations is in lexicographic order, so a base-101 code is sorted and searchsorted
# maps any split straight to its row.
_BASE = (c.TOTAL_SOLDIERS + 1) ** np.arange(c.N_CASTLES - 1, -1, -1, dtype=np.int64)
_CODES = ALL.astype(np.int64) @ _BASE


def split_index(allocs):
    """Row of each split in all_allocations()."""
    codes = np.atleast_2d(np.asarray(allocs, dtype=np.int64)) @ _BASE
    idx = np.searchsorted(_CODES, codes)
    if not np.array_equal(_CODES[np.minimum(idx, N_ALL - 1)], codes):
        raise ValueError("not a legal allocation")
    return idx


def band_of(rank):
    return next(f"{lo}-{hi}" for lo, hi in BANDS if lo <= rank <= hi)


class Tables(NamedTuple):
    """Per member, every split sorted by predicted score, so a score window is one slice."""
    score: np.ndarray    # (M, N) float32, ascending
    idx: np.ndarray      # (M, N) int32, split row at each sorted position
    lib_w: np.ndarray    # (M, N) float32, the member's fitted share of that split (0 if unsampled)
    log_pi: np.ndarray   # (M,) log member weight
    lib: list            # per member, (sorted split rows, weights) of its fitted library


def build_tables(members, probs):
    m_n = len(members)
    score = np.empty((m_n, N_ALL), np.float32)
    idx = np.empty((m_n, N_ALL), np.int32)
    lib_w = np.empty((m_n, N_ALL), np.float32)
    lib = []
    for k, m in enumerate(members):
        s = c.expected_score(c.win_value(c.marginals(m.allocs, m.weights)), ALL)
        order = np.argsort(s, kind="stable")
        dense = np.zeros(N_ALL)
        np.add.at(dense, split_index(m.allocs), m.weights)
        score[k], idx[k], lib_w[k] = s[order], order, dense[order]
        rows = np.flatnonzero(dense)
        lib.append((rows, dense[rows]))
    log_pi = np.log(np.maximum(np.asarray(probs, dtype=float), 1e-300))
    return Tables(score, idx, lib_w, log_pi, lib)


def known_splits(anchors):
    """(rows, live scores) for splits whose score was measured, from (alloc, score) pairs."""
    if not len(anchors):
        return None
    return split_index([a for a, _ in anchors]), np.array([s for _, s in anchors], dtype=float)


def posterior(tables, s, eps, sigma, known=None):
    """Posterior over splits for board score `s`: (rows, p), with p summing to 1.

    Each member only contributes splits within WINDOW sds of `s`; past that the
    likelihood is below e^-8 of the peak and the floor prior can't make up for it.

    `known` (rows, scores) are splits whose live score was measured. For them the
    likelihood is exact: certain if `s` is their score to 3 dp, impossible otherwise.
    Everything else gets the model's Gaussian likelihood of landing in the same 3-dp bin.
    """
    var = sigma**2 + ROUND_VAR
    half = WINDOW * np.sqrt(var)
    log_bin = np.log(0.001) - 0.5 * np.log(2 * np.pi * var)  # P(score lands in s's 3-dp bin)
    gauss, exact = [], []  # (log weights, split rows) per member
    for k in range(len(tables.log_pi)):
        lo, hi = np.searchsorted(tables.score[k], [s - half, s + half])
        if hi <= lo:
            continue
        pred = tables.score[k, lo:hi].astype(float)
        prior = (1 - eps) * tables.lib_w[k, lo:hi] + eps / N_ALL
        gauss.append((tables.log_pi[k] + np.log(prior) - 0.5 * (s - pred) ** 2 / var + log_bin,
                      tables.idx[k, lo:hi]))
    if known is not None:
        here = known[0][np.abs(known[1] - s) < 5e-4]  # splits measured at exactly this score
        for k, (lib_rows, lib_vals) in enumerate(tables.lib):
            w = np.zeros(len(here))
            if len(here) and len(lib_rows):
                pos = np.minimum(np.searchsorted(lib_rows, here), len(lib_rows) - 1)
                hit = lib_rows[pos] == here
                w[hit] = lib_vals[pos[hit]]
            exact.append((tables.log_pi[k] + np.log((1 - eps) * w + eps / N_ALL), here))
    parts = [l for l, _ in gauss + exact if len(l)]
    if not parts:
        return np.empty(0, np.int64), np.empty(0)
    top = max(l.max() for l in parts)
    dense = np.zeros(N_ALL)
    for l, i in gauss:
        dense[i] += np.exp(l - top)  # no repeats within one member, so += is exact
    if known is not None:
        dense[known[0]] = 0.0  # a measured split's model mass is replaced by its exact one
        for l, i in exact:
            dense[i] += np.exp(l - top)
    rows = np.flatnonzero(dense)
    if len(rows) == 0:
        return np.empty(0, np.int64), np.empty(0)
    p = dense[rows]
    return rows, p / p.sum()


def _weighted_quantile(x, p, q):
    order = np.argsort(x, kind="stable")
    cdf = np.cumsum(p[order])
    return x[order][np.minimum(np.searchsorted(cdf, q), len(x) - 1)]


def _intervals(splits, p):
    return [tuple(_weighted_quantile(splits[:, k], p, [0.05, 0.95])) for k in range(c.N_CASTLES)]


def summarize(rows, p, top=3):
    """What a posterior says about one entry."""
    if len(rows) == 0:
        return None
    best = np.argsort(-p)[:top]
    splits = ALL[rows].astype(np.int64)
    map_split = splits[best[0]]
    near = np.isin(rows, split_index(np.vstack([map_split, rb.neighbors(map_split)])))
    nz = p[p > 0]
    out = {"map": fmt(map_split), "p_map": float(p[best[0]]),
           "p_within_1": float(p[near].sum()),
           "p_concede5": float(p[splits[:, 4] <= CONCEDE5].sum()),
           "n_eff": float(np.exp(-(nz * np.log(nz)).sum())),
           "runner_up": "; ".join(f"{fmt(splits[b])} ({p[b]:.2f})" for b in best[1:])}
    for k, (lo, hi) in enumerate(_intervals(splits, p)):
        out[f"c{k + 1}_90"] = f"{lo}-{hi}"
    return out


def score_truth(tables, s, truth, known=None):
    """At every (eps, sigma) grid point: how much mass the true split got, and whether the
    summaries a reader would rely on (MAP, one-move neighborhood, 90% intervals, the
    castle-5 family call) held up."""
    truth = np.asarray(truth, dtype=np.int64)
    t = split_index(truth)[0]
    near_truth = split_index(np.vstack([truth, rb.neighbors(truth)]))
    out = []
    for eps, sigma in itertools.product(EPS_GRID, SIGMA_GRID):
        rows, p = posterior(tables, s, eps, sigma, known)
        row = {"eps": eps, "sigma": sigma, "truth_concede5": bool(truth[4] <= CONCEDE5)}
        if len(rows) == 0:
            out.append({**row, "p_true": 0.0, "log_p_true": np.log(1e-12), "map_correct": False,
                        "p_map": 0.0, "true_within_1_of_map": False, "castle_cover": 0.0,
                        "all_castles_cover": False, "ci_width": np.nan, "p_concede5": np.nan,
                        "n_eff": np.nan})
            continue
        hit = np.flatnonzero(rows == t)
        p_true = float(p[hit[0]]) if len(hit) else 0.0
        best = int(np.argmax(p))
        splits = ALL[rows].astype(np.int64)
        iv = _intervals(splits, p)
        cover = [lo <= truth[k] <= hi for k, (lo, hi) in enumerate(iv)]
        nz = p[p > 0]
        out.append({**row, "p_true": p_true, "log_p_true": float(np.log(max(p_true, 1e-12))),
                    "map_correct": bool(rows[best] == t), "p_map": float(p[best]),
                    "true_within_1_of_map": bool(rows[best] in near_truth),
                    "castle_cover": float(np.mean(cover)), "all_castles_cover": bool(all(cover)),
                    "ci_width": float(np.mean([hi - lo for lo, hi in iv])),
                    "p_concede5": float(p[splits[:, 4] <= CONCEDE5].sum()),
                    "n_eff": float(np.exp(-(nz * np.log(nz)).sum()))})
    return out


# ---------------------------------------------------------------------------
# Validation jobs (each runs in its own process: one ensemble build per job)
# ---------------------------------------------------------------------------

def _ranks(board):
    """Site-style rank of every entry: 1 + entries with a strictly higher score."""
    return 1 + np.searchsorted(np.sort(-board), -board, side="left")


def _probe_walk(start, field, n, rng):
    """Test entries the way we ran them: from a posted split, try a random one-soldier
    move, and keep it as the new base only if it measures better."""
    wv = c.win_value(c.marginals(field))
    base, base_s, out = start, c.expected_score(wv, start[None])[0], []
    while len(out) < n:
        nb = rb.neighbors(base)
        step = nb[rng.integers(len(nb))]
        s = c.expected_score(wv, step[None])[0]
        out.append(step)
        if s > base_s:
            base, base_s = step, s
    return np.array(out)


def synthetic_job(seed):
    """A synthetic board with known splits, scored by rank band at every grid point.

    The true field comes from validate.synthetic_existing, whose generators the model
    doesn't share. Anchors are shaped like ours: the five corners, a posted leader, our own
    entry, two GitHub splits, and N_PROBES test entries climbing from the leader.
    """
    rng = np.random.default_rng(seed)
    existing, leader = vd.synthetic_existing(rng)
    board = np.round(vd.scores_against(existing, existing), 3)
    own = existing[rng.choice(np.argsort(board)[int(0.6 * len(board)):int(0.9 * len(board))])]
    github = existing[rng.choice(np.argsort(board)[int(0.1 * len(board)):int(0.7 * len(board))], 2)]
    probes = _probe_walk(leader, existing, N_PROBES, rng)
    known = np.unique(np.concatenate([vd.CORNERS, [leader, own], github, probes]), axis=0)
    # Probes join the board, so every score (theirs included) is against field + probes.
    field = np.concatenate([existing, probes])
    board = np.round(vd.scores_against(field, existing), 3)
    anchors = [(a, float(s)) for a, s in zip(known, np.round(vd.scores_against(field, known), 3))]
    members = rb.build_ensemble(anchors, board, np.random.default_rng(seed + 1))
    tables = build_tables(members, rb.member_weights(members))

    ranks = _ranks(board)
    # Entries playing an anchor's split are matched exactly by construction; leave them out.
    eligible = ~np.isin(split_index(existing), split_index(known))
    out = []
    for b_lo, b_hi in BANDS:
        pool = np.flatnonzero((ranks >= b_lo) & (ranks <= b_hi) & eligible)
        for e in rng.choice(pool, min(PER_BAND, len(pool)), replace=False):
            for r in score_truth(tables, float(board[e]), existing[e], known_splits(anchors)):
                out.append({"source": "synthetic", "seed": seed, "band": f"{b_lo}-{b_hi}",
                            "rank": int(ranks[e]), "score": float(board[e]), **r})
    return out


def loo_job(k, anchors, board, public_labels):
    """Hold out real anchor k: refit without it, then infer its split from its live score."""
    label, alloc, score = anchors[k]
    saved = bf.KNOWN_SPLITS
    try:
        # A private split (our own entries) can't sit in the prior's focal set while it is
        # the thing being inferred. Public splits stay: anyone could have copied them.
        if label not in public_labels:
            bf.KNOWN_SPLITS = {n: s for n, s in saved.items() if list(s) != list(alloc)}
        rest = [(a, s) for i, (_, a, s) in enumerate(anchors) if i != k]
        members = rb.build_ensemble(rest, board, np.random.default_rng(SEED + k))
    finally:
        bf.KNOWN_SPLITS = saved  # worker processes are reused across jobs
    tables = build_tables(members, rb.member_weights(members))
    rank = int(1 + (board > score).sum())
    return [{"source": "real anchor (held out)", "anchor": fmt(alloc), "band": band_of(rank),
             "public": label in public_labels, "rank": rank, "score": score, **r}
            for r in score_truth(tables, score, alloc, known_splits(rest))]


def _real_board():
    data = sl.read_snapshot(sl.snapshot_files()[-1])
    g1 = pd.DataFrame(data["g1"])
    anchors = load_anchors(g1, quiet=True)
    table = pd.read_csv(ROOT / "data" / "anchors.csv", dtype=str)
    private = table["source"].str.contains("friend|my submission", case=False)
    return data["generated_at"], g1["p"].to_numpy(), anchors, set(table.loc[~private, "label"])


def validate(workers, n_synth):
    generated_at, board, anchors, public = _real_board()
    t0 = time.time()
    rows = []
    out = RESULTS / "inference_validation.csv"
    RESULTS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # Real anchors first: they're the out-of-sample test and the cheaper jobs.
        futs = {pool.submit(loo_job, k, anchors, board, public): f"anchor {fmt(anchors[k][1])}"
                for k in range(len(anchors))}
        futs.update({pool.submit(synthetic_job, 5000 + i): f"synthetic {i}" for i in range(n_synth)})
        for n, fut in enumerate(as_completed(futs), 1):
            rows += fut.result()
            print(f"[{time.time() - t0:5.0f}s] {n}/{len(futs)} {futs[fut]}", flush=True)
            # Save as we go, so a long run that dies partway still leaves usable results.
            pd.DataFrame(rows).assign(board=generated_at).to_csv(out, index=False)
    return pd.DataFrame(rows).assign(board=generated_at)


def choose(df):
    """Per rank band, the (eps, sigma) with the best mean log score of the true split on
    synthetic boards. Log score is a proper scoring rule: it rewards mass on the truth and
    punishes confident misses, which is what calibrated confidence has to mean."""
    synth = df[df["source"] == "synthetic"]
    grid = synth.groupby(["band", "eps", "sigma"])["log_p_true"].mean()
    best = grid.groupby(level="band").idxmax()
    return pd.DataFrame([{"band": b, "eps": e, "sigma": s} for b, e, s in best]), grid


def at_params(df, params):
    m = df.merge(params, on="band", suffixes=("", "_chosen"))
    return m[np.isclose(m["eps"], m["eps_chosen"]) & np.isclose(m["sigma"], m["sigma_chosen"])].copy()


def report(df, params):
    at = at_params(df, params)
    at["family_ok"] = (at["p_concede5"] > 0.5) == at["truth_concede5"]
    at["band_lo"] = at["band"].str.split("-").str[0].astype(int)
    agg = dict(n=("p_true", "size"), top1=("map_correct", "mean"), confidence=("p_map", "mean"),
               within_1=("true_within_1_of_map", "mean"), p_true=("p_true", "mean"),
               castle_cover_90=("castle_cover", "mean"), ci_width=("ci_width", "mean"),
               family_acc=("family_ok", "mean"), n_eff=("n_eff", "median"))
    synth = at[at["source"] == "synthetic"]
    by_band = synth.groupby(["band_lo", "band"]).agg(**agg).droplevel(0)
    return by_band, at[at["source"] != "synthetic"]


# ---------------------------------------------------------------------------
# Real board
# ---------------------------------------------------------------------------

def infer(params, top_n=300, min_cluster=5):
    """Posterior summaries for the top `top_n` entries' distinct scores and every exact-tie
    cluster of at least `min_cluster` entries, each at its band's (eps, sigma). Rank and
    score only; no names."""
    generated_at, board, anchors, _ = _real_board()
    pairs = [(a, s) for _, a, s in anchors]
    members = rb.build_ensemble(pairs, board, np.random.default_rng(SEED))
    tables = build_tables(members, rb.member_weights(members))
    measured = known_splits(pairs)
    by_band = params.set_index("band")
    vals, counts = np.unique(board, return_counts=True)
    cutoff = np.sort(board)[::-1][min(top_n, len(board)) - 1]
    known = {}
    for _, a, s in anchors:
        known.setdefault(round(s, 3), []).append(fmt(a))
    out = []
    for v, n in sorted(zip(vals, counts), reverse=True):
        if v < cutoff and n < min_cluster:
            continue
        rank = int(1 + (board > v).sum())
        eps, sigma = by_band.loc[band_of(rank), ["eps", "sigma"]]
        summ = summarize(*posterior(tables, float(v), float(eps), float(sigma), measured))
        if summ:
            out.append({"rank": rank, "score": float(v), "n_tied": int(n),
                        "known_anchor": " / ".join(known.get(round(float(v), 3), [])), **summ})
    df = pd.DataFrame(out)
    INFERRED.mkdir(parents=True, exist_ok=True)
    df.to_csv(INFERRED / "game1_splits.csv", index=False)
    print(f"board {generated_at}: {len(df)} distinct scores -> {INFERRED / 'game1_splits.csv'}")
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--infer", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--synthetic", type=int, default=8)
    args = ap.parse_args(argv)
    pd.set_option("display.width", 220)
    if args.validate:
        df = validate(args.workers, args.synthetic)
    else:
        df = pd.read_csv(RESULTS / "inference_validation.csv")
    params, grid = choose(df)
    params.to_csv(PARAMS, index=False)
    print("\nmean log P(true split) on synthetic boards, by band (rows) and (eps, sigma) (cols):")
    print(grid.unstack(["eps", "sigma"]).round(1).to_string())
    by_band, real = report(df, params)
    print("\nchosen per band:\n" + params.to_string(index=False))
    print("\nsynthetic boards, by rank band:")
    print(by_band.round(3).to_string())
    print("\nreal anchors, each held out (at their band's parameters):")
    print(real[["anchor", "public", "rank", "score", "map_correct", "p_map", "p_true",
                "true_within_1_of_map", "castle_cover", "ci_width", "p_concede5", "n_eff"]]
          .sort_values("rank").round(3).to_string(index=False))
    if args.infer:
        print(infer(params).head(40).to_string(index=False))


if __name__ == "__main__":
    main()
