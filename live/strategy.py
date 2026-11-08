"""Game 1 pick from the newest leaderboard snapshot, ranked by mean score across worlds.

Scores alone don't identify the field: on synthetic boards with a known answer, a fit to
the score histogram picked splits that lost 1-2.5 points (board_fit.synthetic_board).
Known (split, score) pairs do. Each anchor in data/anchors.csv is a board entry whose
split is known; every run reads its *current* score from the newest snapshot (scores
drift as people join) and hands the anchors to robust.run, which builds the field
ensemble, samples hidden late movers and ranks candidates by mean score.

The single-fit best response (one prior, tilted to the anchors) is logged alongside as
a baseline.

    .venv/bin/python -m live.strategy --once
    .venv/bin/python -m live.strategy --every 300 --until 2026-11-09T23:30-05:00
"""

import argparse
import csv
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from live import board_fit as bf
from sim import castles as c
from live import newcomers as nc
from live import probes
from live import robust
from live import scrape_leaderboard as sl
# Anchor loading lives in anchors.py; re-exported because game2 and the tests use st.*.
from live.anchors import ANCHORS, DATA, _baseline_ids, anchor_rows, fmt, load_anchors, resolve_id  # noqa: F401

OUT = DATA / "strategy"
SEED = 2026  # fixed so consecutive runs differ only because the board or anchors changed
RUN_FIELDS = ["run_at_utc", "generated_at", "n_g1", "n_anchors", "concede5_weight", "swing_pick",
              "swing_p1", "swing_ptop10", "swing_mean", "swing_rank", "second", "second_p1",
              "single_fit_pick", "board_top", "cohort_k", "cohort_trackers", "cohort_c4_48"]

log = sl.log


def tilted_field(anchors, rng, rounds=2):
    """Prior library tilted to the anchors. Later rounds rebuild the sharp players as
    noisy best responses to the tilted field, since they respond to the real field."""
    pairs = [(a, s) for _, a, s in anchors]
    lib = bf.build_library(rng)
    w = bf.tilt_to_anchors(lib.prior, lib.allocs, pairs, sweeps=50)
    for _ in range(rounds - 1):
        lib = bf.build_library(rng, sharp_marg=c.marginals(lib.allocs, w))
        w = bf.tilt_to_anchors(lib.prior, lib.allocs, pairs, sweeps=50)
    return lib, w


def processed(path=OUT / "runs.csv"):
    if not path.exists():
        return set()
    return set(pd.read_csv(path, dtype=str)["generated_at"])


def run_once(force=False):
    files = sl.snapshot_files()
    if not files:
        log("no snapshots yet")
        return None
    data = sl.read_snapshot(files[-1])
    gen = data["generated_at"]
    if not force and gen in processed():
        return None  # quiet: the loop polls often and most polls find nothing new

    t0 = time.time()
    g1 = pd.DataFrame(data["g1"])
    board = g1["p"].to_numpy()
    anchors = load_anchors(g1)
    pairs = [(a, s) for _, a, s in anchors]
    # The single-fit best response is kept as a baseline to compare the pick against.
    lib, w = tilted_field(anchors, np.random.default_rng(SEED))
    single_fit, _ = c.best_response(c.win_value(c.marginals(lib.allocs, w)))
    # Newcomer cohort, read from anchor score changes since the first snapshot.
    w_top = bf.tilt_to_board_top(lib.prior, lib.allocs, pairs, float(board.max()), sweeps=50)
    cohort = nc.track(anchor_rows(g1), lib.allocs, w_top, files, np.random.default_rng(SEED))
    use_cohort = cohort.k >= nc.MIN_COHORT
    members, probs, worlds, res = robust.run(
        pairs, board, np.random.default_rng(SEED), late=nc.late_prior(cohort, robust.LATE),
        cohort=cohort.field if use_cohort else None)
    order = robust.rank_candidates(res)
    trackers = sum(cohort.shares.get(k, 0.0) for k in ("L1", "L2", "L3"))
    concede5 = float(sum(p for m, p in zip(members, probs) if m.br[4] < 30))

    best, second = order[0], order[1]
    row = {"run_at_utc": f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}", "generated_at": gen,
           "n_g1": len(board), "n_anchors": len(anchors), "concede5_weight": round(concede5, 3),
           "swing_pick": fmt(res.allocs[best]), "swing_p1": round(float(res.p_first[best]), 3),
           "swing_ptop10": round(float(res.p_top10[best]), 3),
           "swing_mean": round(float(res.mean_score[best]), 3),
           "swing_rank": round(float(res.mean_rank[best]), 1),
           "second": fmt(res.allocs[second]), "second_p1": round(float(res.p_first[second]), 3),
           "single_fit_pick": fmt(single_fit), "board_top": float(board.max()),
           "cohort_k": cohort.k, "cohort_trackers": round(trackers, 3),
           "cohort_c4_48": round(cohort.castle4.get(48, float("nan")), 3)}
    OUT.mkdir(parents=True, exist_ok=True)
    runs = OUT / "runs.csv"
    new = not runs.exists()
    with open(runs, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow(row)
    write_report(row, anchors, members, probs, res, order, cohort, use_cohort)
    log(f"strategy: pick {row['swing_pick']} mean {row['swing_mean']:.3f} P(top10) "
        f"{row['swing_ptop10']:.2f} | concede-5 weight {concede5:.2f} | {len(anchors)} anchors | "
        f"newcomers {cohort.k} ({trackers:.0%} trackers{'' if use_cohort else ', not used yet'}) | "
        f"{time.time() - t0:.0f}s")
    return row


def write_report(row, anchors, members, probs, res, order, cohort, use_cohort, n=10):
    lines = [
        f"# Game 1 pick, board of {row['generated_at']}",
        "",
        f"Board: {row['n_g1']} entries, top {row['board_top']:.3f}. {row['n_anchors']} anchors.",
        f"Weight on the concede-castle-5 family: {row['concede5_weight']:.2f}.",
        "",
        "| # | allocation | P(#1) | P(top 10) | mean score | mean rank | wins when trackers are L1/L2/L3 | sharp share |",
        "|---|---|---|---|---|---|---|---|",
        *[f"| {i + 1} | {fmt(res.allocs[k])} | {res.p_first[k]:.3f} | {res.p_top10[k]:.3f} | "
          f"{res.mean_score[k]:.3f} | {res.mean_rank[k]:.0f} | "
          f"{' / '.join(f'{x:.2f}' for x in res.bet_levels[k])} | {res.bet_sharp[k]:.2f} |"
          for i, k in enumerate(order[:n])],
        "",
        "| ensemble member | weight | best response | leave-one-out MAE |",
        "|---|---|---|---|",
        *[f"| {m.name} | {p:.2f} | {fmt(m.br)} | {np.abs(m.loo).mean():.2f} |"
          for m, p in sorted(zip(members, probs), key=lambda x: -x[1])],
        "",
        "| anchor | split | board score |",
        "|---|---|---|",
        *[f"| {lbl} | {fmt(a)} | {s:.3f} |" for lbl, a, s in anchors],
        "",
        f"## Newcomers since the first snapshot: {cohort.k} "
        f"({'drives the late-entrant model' if use_cohort else 'too few to use yet'})",
        "",
        "Fitted mix: " + ", ".join(f"{k} {v:.2f} ± {cohort.share_sd.get(k, 0):.2f}"
                                   for k, v in cohort.shares.items()) + ".",
        "Share of newcomers sending exactly N to castle 4: "
        + ", ".join(f"{x}: {v:.3f}" for x, v in cohort.castle4.items()) + ".",
        "",
        "| anchor | split | avg points vs newcomers | ± |",
        "|---|---|---|---|",
        *[f"| {r.label} | {fmt(r.alloc)} | {r.mean_payoff:.2f} | {r.sd:.2f} |" for r in cohort.reads],
        "",
        *probes.report_lines(),
    ]
    (OUT / "latest.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="run on the newest snapshot and exit")
    mode.add_argument("--every", type=int, metavar="SECONDS", help="interval for loop mode")
    ap.add_argument("--until", help="stop time for loop mode, ISO 8601 with offset")
    ap.add_argument("--force", action="store_true", help="rerun even if the snapshot was processed")
    args = ap.parse_args(argv)
    if args.once:
        run_once(force=args.force)
        return 0
    until = datetime.fromisoformat(args.until or "")
    if until.tzinfo is None:
        ap.error("--until needs a UTC offset")
    start, k = time.time(), 0
    while datetime.now(timezone.utc) < until:
        try:
            run_once()
        except Exception as exc:  # a bad run must not end the loop before the deadline
            log(f"strategy run failed: {exc!r}")
        k += 1
        time.sleep(max(0.0, start + k * args.every - time.time()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
