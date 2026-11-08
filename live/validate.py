"""Step E: does the swing-for-#1 pipeline beat simpler picks in worlds where we know the truth?

Each trial builds a synthetic existing field from generators the model does not share,
gives every method only what we really have (board scores and nine anchors shaped like
ours), then adds hidden late entrants, including anchor-tracking sharp players at
several thinking levels, and ranks every method's pick against the true final field.

    .venv/bin/python -m live.validate --trials 12   # -> results/live/validation.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from live import board_fit as bf
from sim import castles as c
from live import robust as rb

OUT = Path(__file__).resolve().parents[1] / "results" / "live" / "validation.csv"
N_EXISTING = 5800
CORNERS = np.eye(5, dtype=np.int64) * 100
CHATBOT = np.array([7, 13, 20, 27, 33])


def scores_against(field, allocs):
    return c.expected_score(c.win_value(c.marginals(field)), np.atleast_2d(allocs))


def synthetic_existing(rng):
    """True existing field and the splits that were 'published' about it."""
    tau = rng.uniform(1.0, 3.0)
    lam = rng.uniform(2.0, 10.0)
    shares = rng.dirichlet([4, 3, 1])  # habit, shallow level-k, equilibrium
    n = (shares * N_EXISTING * 0.9).astype(int)
    habit = c.level0_sample("focal", n[0], rng)
    levelk = bf.level_k_samples(c.marginals(habit), tau, lam, n[1], rng, max_level=4)
    eq = bf._equilibrium_plays()
    crowd = np.concatenate([habit, levelk, eq[rng.integers(0, len(eq), n[2])]])
    wv = c.win_value(c.marginals(crowd))
    probers = c.logit_sample(wv, 50.0, int(0.03 * N_EXISTING), rng)
    leader = c.logit_sample(wv, 15.0, 1, rng)[0]  # a strong split someone posts publicly
    copies = np.repeat(np.array([leader, [0, 0, 0, 0, 100], [20] * 5, CHATBOT]),
                       [int(rng.integers(18, 30)), 35, 25, 30], axis=0)
    field = np.concatenate([crowd, probers, copies])
    top = c.best_response(c.win_value(c.marginals(field)))[0]  # probers who found the max
    field = np.concatenate([field, np.repeat(top[None], int(rng.integers(6, 14)), axis=0)])
    return field[:N_EXISTING] if len(field) > N_EXISTING else field, leader


def synthetic_late(existing, public, rng):
    """Hidden late entrants: plain + pasted public splits + anchor trackers at levels 1-4."""
    count = int(rng.integers(100, 600))
    n_sharp = int(rng.uniform(0.15, 0.6) * count)
    n_plain = count - n_sharp
    plain = existing[rng.integers(0, len(existing), int(0.8 * n_plain))]
    copies = public[rng.integers(0, len(public), n_plain - len(plain))]
    seen = c.marginals(existing)  # trackers in the truth see the field well
    lam = float(rng.choice([20.0, 60.0, 200.0]))
    out, view = [plain, copies], seen
    for n_k in rng.multinomial(n_sharp, rng.dirichlet([1, 1, 1, 0.5])):  # L1..L4
        out.append(c.logit_sample(c.win_value(view), lam, n_k, rng))
        br, _ = c.best_response(c.win_value(view))
        view = (len(existing) * seen + n_sharp * c.marginals(br)) / (len(existing) + n_sharp)
    return np.concatenate(out)


def final_rank(pick, final):
    """Rank of `pick` among `final` + itself, by the site's 3-dp scores (ties share a rank)."""
    everyone = np.concatenate([final, pick[None]])
    s = np.round(scores_against(everyone, everyone), 3)
    return int(1 + (s[:-1] > s[-1]).sum()), float(s[-1])


def trial(seed, n_worlds):
    rng = np.random.default_rng(seed)
    existing, leader = synthetic_existing(rng)
    board = np.round(scores_against(existing, existing), 3)
    own = existing[rng.choice(np.argsort(board)[int(0.6 * len(board)):int(0.9 * len(board))])]
    github = existing[rng.choice(np.argsort(board)[int(0.1 * len(board)):int(0.7 * len(board))], 2)]
    known = np.concatenate([CORNERS, [leader, own], github])
    anchors = [(a, float(s)) for a, s in zip(known, np.round(scores_against(existing, known), 3))]
    public = np.concatenate([[leader, own], github, [CHATBOT]])

    picks = {}
    t = time.time()
    _, _, _, res = rb.run(anchors, board, np.random.default_rng(seed + 1), n_worlds=n_worlds,
                          public=public)
    picks["swing (P#1)"] = res.allocs[rb.rank_candidates(res)[0]]
    picks["swing (mean score)"] = res.allocs[np.argmax(res.mean_score)]
    swing_secs = time.time() - t

    lib = bf.build_library(np.random.default_rng(seed + 2))
    w = bf.tilt_to_anchors(lib.prior, lib.allocs, anchors, sweeps=50)
    picks["single fit BR"] = c.best_response(c.win_value(c.marginals(lib.allocs, w)))[0]
    w_top = bf.tilt_to_board_top(lib.prior, lib.allocs, anchors, board.max(), sweeps=50)
    wv_top = c.win_value(c.marginals(lib.allocs, w_top))
    picks["board-top fit BR (= an L1 tracker)"] = c.best_response(wv_top)[0]
    picks["copy posted leader"] = leader
    nb = rb.neighbors(leader)
    picks["posted leader, best +1 tweak"] = nb[np.argmax(c.expected_score(wv_top, nb))]

    late = synthetic_late(existing, public, rng)
    final = np.concatenate([existing, late])
    picks["oracle: BR to true existing field"] = c.best_response(
        c.win_value(c.marginals(existing)))[0]
    picks["oracle: BR to true final field"] = c.best_response(c.win_value(c.marginals(final)))[0]

    rows = []
    for name, pick in picks.items():
        rank, score = final_rank(np.asarray(pick), final)
        rows.append({"seed": seed, "method": name, "pick": "-".join(map(str, pick)),
                     "rank": rank, "score": score, "n_late": len(late),
                     "swing_secs": round(swing_secs, 1)})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--worlds", type=int, default=200)
    args = ap.parse_args()
    rows = []
    for k in range(args.trials):
        rows += trial(1000 + k, args.worlds)
        df = pd.DataFrame(rows)
        df.to_csv(OUT, index=False)  # after every trial, so a partial run is still usable
        print(f"trial {k + 1}/{args.trials} done", flush=True)
    summary = df.groupby("method")["rank"].agg(
        mean_rank="mean", median_rank="median", first=lambda r: (r == 1).mean(),
        top10=lambda r: (r <= 10).mean(), top50=lambda r: (r <= 50).mean())
    print(summary.sort_values("median_rank").round(2).to_string())


if __name__ == "__main__":
    main()
