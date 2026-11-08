import gzip
import json

import numpy as np
import pytest

from live import probes


def write_snapshot(path, g1, g2):
    with gzip.open(path, "wb") as f:
        f.write(json.dumps({"g1": g1, "g2": g2, "generated_at": "t"}).encode())
    return path


def entry(id_, p):
    return {"id": id_, "f": "X", "l": "Y", "s": "U", "p": p, "r": 1}


def test_gaps_use_latest_g1_and_averaged_g2(tmp_path):
    anchors = tmp_path / "anchors.csv"
    anchors.write_text("board_id,game,label,alloc,source\n"
                       "a,g1,base,2-8-35-48-7,s\nb,g1,up,2-8-35-49-6,s\n"
                       "a,g2,base2,18-8-2-2-70,s\nb,g2,var2,22-8-0-0-70,s\n")
    files = [write_snapshot(tmp_path / "1.json.gz", [entry("a", 9.0), entry("b", 9.5)],
                            [entry("a", 5.00), entry("b", 5.10)]),
             write_snapshot(tmp_path / "2.json.gz", [entry("a", 9.205), entry("b", 9.136)],
                            [entry("a", 5.02), entry("b", 5.14)])]
    spec = [("g1", "c4 48->49", "2-8-35-48-7", "2-8-35-49-6"),
            ("g2", "scraps", "18-8-2-2-70", "22-8-0-0-70"),
            ("g1", "pending", "2-8-35-48-7", "2-8-35-47-8")]
    g1, g2, pending = probes.comparisons(files, spec, path=anchors)
    assert g1.gap == pytest.approx(9.136 - 9.205) and g1.sd == probes.G1_GAP_SD
    assert g2.gap == pytest.approx(5.12 - 5.01)  # averaged over both boards
    assert pending.status == "waiting for 2-8-35-47-8" and np.isnan(pending.gap)


def test_republished_g2_board_counts_once(tmp_path):
    anchors = tmp_path / "anchors.csv"
    anchors.write_text("board_id,game,label,alloc,source\na,g2,base2,18-8-2-2-70,s\n")
    g1 = [entry("a", 9.0)]
    files = [write_snapshot(tmp_path / "1.json.gz", g1, [entry("a", 5.00)]),
             write_snapshot(tmp_path / "2.json.gz", g1, [entry("a", 5.06)]),
             write_snapshot(tmp_path / "3.json.gz", g1, [entry("a", 5.06)])]  # no one joined
    [r] = probes.reads("g2", files, path=anchors).values()
    assert r.reads == 2 and r.mean == pytest.approx(5.03)
