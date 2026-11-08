"""Snapshot the public Castle Challenge leaderboard on a schedule.

The leaderboard page loads one JSON file holding every entry's name, school, score and rank
for both games (no allocations). Scores are recomputed against the whole pool each time the
file is regenerated, so a time series of snapshots shows how the field moves as new
applicants arrive.

    .venv/bin/python -m live.scrape_leaderboard --once
    .venv/bin/python -m live.scrape_leaderboard --every 900 --until 2026-11-10T09:00-05:00

Raw snapshots land in data/snapshots/ (gitignored: they hold thousands of students' names).
"""

import argparse
import csv
import gzip
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = "https://pokerbot.sequoiacap.com/castle_challenge/data/leaderboard.json"
USER_AGENT = "castle_game leaderboard logger (personal project; polls every 15 min)"
SNAP_DIR = Path(__file__).resolve().parents[1] / "data" / "snapshots"
INDEX_FIELDS = ["fetched_at_utc", "generated_at", "n_g1", "n_g2", "g1_top", "g1_median",
                "g2_top", "g2_median", "new_g1", "new_g2", "file"]
GAMES = ("g1", "g2")


def log(msg):
    # flush so the nohup'd log file is readable while the loop is running
    print(f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {msg}", flush=True)


def parse(raw):
    """Decode a leaderboard payload, failing loudly if the shape changed under us."""
    data = json.loads(raw)
    missing = [k for k in (*GAMES, "generated_at") if k not in data]
    if missing:
        raise ValueError(f"leaderboard JSON missing keys: {missing}")
    for g in GAMES:
        if not isinstance(data[g], list) or (data[g] and not {"id", "p", "r"} <= data[g][0].keys()):
            raise ValueError(f"unexpected entry format in {g}")
    return data


def fetch(url=URL, timeout=30, retries=3):
    """Return (raw bytes, parsed dict). Retries transient failures with backoff."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Cache-Control": "no-cache"})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return raw, parse(raw)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            if attempt == retries:
                raise
            log(f"fetch attempt {attempt} failed ({exc}); retrying")
            time.sleep(10 * attempt)


def snapshot_name(generated_at):
    # '2026-11-07T21:06:29Z' -> '20261107T210629Z.json.gz'; sorts chronologically by name
    return generated_at.replace("-", "").replace(":", "") + ".json.gz"


def snapshot_files(snap_dir=SNAP_DIR):
    return sorted(Path(snap_dir).glob("*.json.gz"))


def read_snapshot(path):
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


def _summary(data, prev, fetched_at, file_name):
    row = {"fetched_at_utc": fetched_at, "generated_at": data["generated_at"], "file": file_name}
    for g in GAMES:
        scores = [e["p"] for e in data[g]]
        row[f"n_{g}"] = len(scores)
        row[f"{g}_top"] = max(scores, default="")
        row[f"{g}_median"] = statistics.median(scores) if scores else ""
        # Entries not seen in the previous snapshot = new applicants since then.
        prev_ids = {e["id"] for e in prev[g]} if prev else set()
        row[f"new_{g}"] = sum(e["id"] not in prev_ids for e in data[g]) if prev else ""
    return row


def save_snapshot(raw, data, snap_dir=SNAP_DIR, fetched_at=None):
    """Write the snapshot and an index row. Returns the row, or None if already saved.

    Dedupe key is generated_at: the server only changes the file when it regenerates,
    so a repeat poll between regenerations is a no-op. That also makes a restart safe.
    """
    snap_dir = Path(snap_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    path = snap_dir / snapshot_name(data["generated_at"])
    if path.exists():
        return None

    previous = snapshot_files(snap_dir)
    prev = read_snapshot(previous[-1]) if previous else None

    # Write to a temp name and rename so a crash never leaves a half-written snapshot
    # that would then be mistaken for a complete one by the dedupe check.
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb") as f:
        f.write(raw)
    tmp.rename(path)

    fetched_at = fetched_at or f"{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"
    row = _summary(data, prev, fetched_at, path.name)
    index = snap_dir / "index.csv"
    new_index = not index.exists()
    with open(index, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new_index:
            writer.writeheader()
        writer.writerow(row)
    return row


def poll_once(snap_dir=SNAP_DIR, fetch_fn=fetch):
    raw, data = fetch_fn()
    row = save_snapshot(raw, data, snap_dir)
    if row is None:
        log(f"unchanged (generated_at {data['generated_at']})")
    else:
        log(f"saved {row['file']}: g1 n={row['n_g1']} (+{row['new_g1']}) top={row['g1_top']} "
            f"median={row['g1_median']} | g2 n={row['n_g2']} (+{row['new_g2']}) "
            f"top={row['g2_top']}")
    return row


def run_loop(every, until, snap_dir=SNAP_DIR):
    """Poll every `every` seconds until `until`. A failed poll is logged, never fatal."""
    pid_file = Path(snap_dir).parent / "scraper.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    log(f"polling every {every}s until {until.isoformat()} (pid {os.getpid()})")
    start = time.time()
    k = 0
    try:
        while datetime.now(timezone.utc) < until:
            try:
                poll_once(snap_dir)
            except Exception as exc:  # keep the loop alive through network or server hiccups
                log(f"poll failed: {exc!r}")
            k += 1
            # Schedule against the start time so slow fetches don't make the cadence drift.
            time.sleep(max(0.0, start + k * every - time.time()))
        log("reached --until; stopping")
    finally:
        pid_file.unlink(missing_ok=True)


def load_snapshots(snap_dir=SNAP_DIR):
    """All snapshots as one long frame: generated_at, game, id, name, school, score, rank."""
    import pandas as pd  # only analysis needs pandas; the scraper loop runs on the stdlib

    rows = []
    for path in snapshot_files(snap_dir):
        data = read_snapshot(path)
        for g in GAMES:
            for e in data[g]:
                rows.append((data["generated_at"], g, e["id"], e.get("n") or f"{e['f']} {e['l']}.",
                             e.get("s", ""), e["p"], e["r"]))
    df = pd.DataFrame(rows, columns=["generated_at", "game", "id", "name", "school", "score", "rank"])
    df["generated_at"] = pd.to_datetime(df["generated_at"])
    return df


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true", help="poll once and exit")
    mode.add_argument("--every", type=int, metavar="SECONDS", help="poll interval for loop mode")
    ap.add_argument("--until", help="stop time for loop mode, ISO 8601 with offset")
    args = ap.parse_args(argv)

    if args.once:
        poll_once()
        return 0
    if not args.until:
        ap.error("--every needs --until")
    until = datetime.fromisoformat(args.until)
    if until.tzinfo is None:
        ap.error("--until needs a UTC offset, e.g. 2026-11-10T09:00-05:00")
    run_loop(args.every, until)
    return 0


if __name__ == "__main__":
    sys.exit(main())
