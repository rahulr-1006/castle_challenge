"""Test-entry results read straight off the board, with no model involved.

Two entries that differ by a soldier or two measure the value of that move exactly: the
gap between their live scores. Game 1 scores are exact up to 3-dp rounding (+-0.001 on a
gap). Game 2 scores carry the site's per-entry simulation noise, so they are averaged over
the last few snapshots and reported with a standard error.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd

from live import anchors as an
from live import scrape_leaderboard as sl

# (game, what the gap measures, split A, split B); gap = score(B) - score(A).
COMPARISONS = [
    ("g1", "castle 4: 48 -> 49, paid by castle 5", "2-8-35-48-7", "2-8-35-49-6"),
    ("g1", "castle 4: 49 -> 50, paid by castle 5", "2-8-35-49-6", "2-8-35-50-5"),
    ("g1", "48/7 -> 47/8 (castle 4 -> castle 5)", "2-8-35-48-7", "2-8-35-47-8"),
    ("g1", "castle 5: 8 -> 9, paid by castle 1", "2-8-35-47-8", "1-8-35-47-9"),
    ("g1", "48/7 -> 47/9 (castles 1 and 4 -> castle 5)", "2-8-35-48-7", "1-8-35-47-9"),
    ("g1", "FAMILY: concede castle 5 -> stack castle 5", "2-8-35-48-7", "4-8-27-3-58"),
    ("g1", "FAMILY vs best measured (AAA)", "5-7-35-47-6", "4-8-27-3-58"),
    ("g1", "AAA: castle 5 6 -> 7, paid by castle 1", "5-7-35-47-6", "4-7-35-47-7"),
    ("g1", "castle 3: 35 -> 36, paid by castle 1", "4-7-35-47-7", "3-7-36-47-7"),
    ("g1", "castle 4: 47 -> 48, paid by castle 2", "4-7-35-47-7", "4-6-35-48-7"),
    ("g2", "castle 5: 70 -> 74, paid by castle 1", "18-8-2-2-70", "14-8-2-2-74"),
    ("g2", "castle 2 -> castle 1 (at 74 on castle 5)", "14-8-2-2-74", "20-2-2-2-74"),
    ("g2", "switch to castles 1+2+3 (model's old top claim)", "14-8-2-2-74", "13-38-49-0-0"),
    ("g2", "castle 5 -2 -> castles 3 and 4 +1 each", "20-2-2-2-74", "20-2-3-3-72"),
    ("g2", "castle 5: 74 -> 78, paid by castle 1", "20-2-2-2-74", "16-2-2-2-78"),
    ("g2", "castle 5: 74 -> 80, paid by the 2s on castles 2-4", "20-2-2-2-74", "20-0-0-0-80"),
]
WINDOW = 6           # snapshots averaged for Game 2
G1_GAP_SD = 0.001    # two 3-dp roundings
G2_NOISE_FLOOR = 0.01


class Read(NamedTuple):
    label: str
    split: str
    mean: float
    sd: float   # standard error of the mean
    reads: int


class Gap(NamedTuple):
    game: str
    what: str
    a: str
    b: str
    score_a: float
    score_b: float
    gap: float
    sd: float
    status: str  # "measured", or which side is still missing


def distinct_boards(game, files, window):
    """The latest board for Game 1 (exact). For Game 2 (noisy), the newest `window` boards
    that differ from the snapshot after them: when nobody joins, the site republishes the
    same Game 2 scores, and counting a repeat would shrink the standard error without
    adding a new simulation."""
    if not files:
        return []
    if game == "g1":
        return [pd.DataFrame(sl.read_snapshot(files[-1])["g1"])]
    out, newer = [], None
    for f in reversed(files):
        board = pd.DataFrame(sl.read_snapshot(f)["g2"])
        key = tuple(board.sort_values("id")["p"])
        if key != newer:
            out.append(board)
            if len(out) == window:
                break
        newer = key
    return out


def reads(game, files=None, window=WINDOW, path=an.ANCHORS):
    """Every anchor's score for `game`, averaged over `distinct_boards`. Returns {split: Read}."""
    files = sl.snapshot_files() if files is None else files
    acc = {}
    for board in distinct_boards(game, files, window):
        for label, alloc, score in an.load_anchors(board, path=path, game=game, quiet=True):
            acc.setdefault(an.fmt(alloc), (label, []))[1].append(score)
    out = {}
    for split, (label, v) in acc.items():
        se = np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
        out[split] = Read(label, split, float(np.mean(v)), float(se), len(v))
    return out


def comparisons(files=None, spec=COMPARISONS, path=an.ANCHORS):
    by_game = {g: reads(g, files, path=path) for g in {s[0] for s in spec}}
    out = []
    for game, what, a, b in spec:
        ra, rb = by_game[game].get(a), by_game[game].get(b)
        missing = [x for x, r in ((a, ra), (b, rb)) if r is None]
        if missing:
            out.append(Gap(game, what, a, b, ra.mean if ra else np.nan, rb.mean if rb else np.nan,
                           np.nan, np.nan, "waiting for " + " and ".join(missing)))
            continue
        if game == "g1":
            sd = G1_GAP_SD
        else:
            se = [r.sd if np.isfinite(r.sd) else G2_NOISE_FLOOR for r in (ra, rb)]
            sd = float(np.hypot(max(se[0], G2_NOISE_FLOOR / 2), max(se[1], G2_NOISE_FLOOR / 2)))
        out.append(Gap(game, what, a, b, ra.mean, rb.mean, rb.mean - ra.mean, sd, "measured"))
    return out


def report_lines(files=None):
    gaps = comparisons(files)
    g2 = sorted(reads("g2", files).values(), key=lambda r: -r.mean)
    lines = ["## Test-entry comparisons (straight from the board, no model)", "",
             "| game | what the gap measures | A | B | score A | score B | B - A | ± |",
             "|---|---|---|---|---|---|---|---|"]
    for g in gaps:
        if g.status == "measured":
            lines.append(f"| {g.game} | {g.what} | {g.a} | {g.b} | {g.score_a:.3f} | {g.score_b:.3f} "
                         f"| **{g.gap:+.3f}** | {g.sd:.3f} |")
        else:
            lines.append(f"| {g.game} | {g.what} | {g.a} | {g.b} | | | {g.status} | |")
    lines += ["", f"## Game 2 splits ranked by measured score (avg of last {WINDOW} boards)", "",
              "| split | who | score | ± | reads |", "|---|---|---|---|---|"]
    lines += [f"| {r.split} | {r.label} | {r.mean:.3f} | {r.sd:.3f} | {r.reads} |" for r in g2]
    return lines + [""]
