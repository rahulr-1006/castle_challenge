"""Anchors: board entries whose split is known (data/anchors.csv), read against a snapshot.

Each row of data/anchors.csv is (board_id, game, label, alloc, source). board_id is an
entry id, or "name:<full name>" for an entry whose id isn't known yet.
"""

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from sim import castles as c
from live import scrape_leaderboard as sl

DATA = Path(__file__).resolve().parents[1] / "data"
ANCHORS = DATA / "anchors.csv"

log = sl.log


def fmt(alloc):
    return "-".join(str(int(x)) for x in alloc)


def parse(text):
    alloc = np.array([int(x) for x in text.split("-")])
    if alloc.sum() != c.TOTAL_SOLDIERS or len(alloc) != c.N_CASTLES or (alloc < 0).any():
        raise ValueError(f"{text} is not a legal allocation")
    return alloc


@functools.lru_cache(maxsize=1)
def _baseline_ids():
    """Ids on the first snapshot: anyone already there can't be a new test entry."""
    files = sl.snapshot_files()
    if not files:
        return frozenset()
    first = sl.read_snapshot(files[0])
    return frozenset(e["id"] for g in ("g1", "g2") for e in first[g])


def resolve_id(key, board):
    """Board id for an anchors.csv key: an id, or "name:<full name>" for entries whose id
    isn't known yet. A full-name match wins; otherwise, since many entries show only a
    first name and last initial, a unique first-name + initial match among entries that
    arrived after the first snapshot."""
    if not key.startswith("name:"):
        return key if key in set(board["id"]) else None
    name = key[5:].strip().lower()
    if "n" in board:
        hits = board[board["n"].fillna("").str.strip().str.lower() == name]
        if len(hits) == 1:
            return hits["id"].iloc[0]
    if not {"f", "l"} <= set(board.columns):
        return None
    first, last = name.rsplit(" ", 1) if " " in name else (name, "")
    new = board[~board["id"].isin(_baseline_ids())]
    hits = new[(new["f"].str.strip().str.lower() == first)
               & (new["l"].str.strip().str.lower().str[:1] == last[:1])]
    return hits["id"].iloc[0] if len(hits) == 1 else None


def load_anchors(board, path=ANCHORS, game="g1", quiet=False):
    """Anchors for `game` present on this board as (label, alloc, current score)."""
    table = pd.read_csv(path, dtype=str)
    table = table[table["game"] == game]
    scores = dict(zip(board["id"], board["p"]))
    out = []
    for row in table.itertuples():
        try:
            alloc = parse(row.alloc)
        except ValueError as exc:
            raise ValueError(f"anchor {row.label}: {exc}") from None
        board_id = resolve_id(row.board_id, board)
        if board_id is None:
            if not quiet:
                log(f"anchor {row.label} ({row.board_id}) not on this board; skipped")
            continue
        out.append((row.label, alloc, float(scores[board_id])))
    return out


def anchor_rows(board, path=ANCHORS, game="g1"):
    """(label, board_id, alloc) for every anchor of `game` whose id resolves on `board`."""
    table = pd.read_csv(path, dtype=str)
    out = []
    for r in table[table["game"] == game].itertuples():
        board_id = resolve_id(r.board_id, board)
        if board_id is not None:
            out.append((r.label, board_id, parse(r.alloc)))
    return out
