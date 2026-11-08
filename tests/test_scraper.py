import csv
import json

import pytest

from live import scrape_leaderboard as sl


def payload(generated_at, g1_ids, g2_ids=("x",)):
    def entries(ids):
        return [{"id": i, "f": "A", "l": "B", "s": "U", "p": 7.0 + n, "r": n + 1}
                for n, i in enumerate(ids)]
    data = {"g1": entries(g1_ids), "g2": entries(g2_ids), "generated_at": generated_at}
    raw = json.dumps(data).encode()
    return raw, sl.parse(raw)


def index_rows(snap_dir):
    with open(snap_dir / "index.csv") as f:
        return list(csv.DictReader(f))


def test_parse_rejects_changed_format():
    with pytest.raises(ValueError):
        sl.parse(b'{"g1": [], "generated_at": "t"}')
    with pytest.raises(ValueError):
        sl.parse(b'{"g1": [{"name": "x"}], "g2": [], "generated_at": "t"}')


def test_dedupe_and_new_entry_counts(tmp_path):
    first = payload("2026-11-07T21:06:29Z", ["a", "b"])
    row = sl.poll_once(tmp_path, fetch_fn=lambda: first)
    assert row["n_g1"] == 2 and row["new_g1"] == ""  # no previous snapshot to diff against

    # Same generated_at: the server hasn't regenerated, so nothing is written.
    assert sl.poll_once(tmp_path, fetch_fn=lambda: first) is None
    assert len(index_rows(tmp_path)) == 1

    second = payload("2026-11-07T21:21:30Z", ["a", "b", "c"])
    row = sl.poll_once(tmp_path, fetch_fn=lambda: second)
    assert row["new_g1"] == 1 and row["new_g2"] == 0

    assert [p.name for p in sl.snapshot_files(tmp_path)] == \
        ["20261107T210629Z.json.gz", "20261107T212130Z.json.gz"]
    assert [r["n_g1"] for r in index_rows(tmp_path)] == ["2", "3"]


def test_load_snapshots(tmp_path):
    sl.poll_once(tmp_path, fetch_fn=lambda: payload("2026-11-07T21:06:29Z", ["a", "b"]))
    sl.poll_once(tmp_path, fetch_fn=lambda: payload("2026-11-07T21:21:30Z", ["a", "b", "c"]))
    df = sl.load_snapshots(tmp_path)
    assert df.groupby("game").size().to_dict() == {"g1": 5, "g2": 2}
    assert df["name"].iloc[0] == "A B."
