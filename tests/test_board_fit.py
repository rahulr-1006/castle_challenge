import numpy as np
import pandas as pd
import pytest

from live import board_fit as bf
from sim import castles as c
from live import strategy


@pytest.fixture(scope="module")
def lib():
    return bf.build_library(np.random.default_rng(0), n=300)


def test_library_is_deduplicated_and_normalized(lib):
    assert len(np.unique(lib.allocs, axis=0)) == len(lib.allocs)
    assert (lib.allocs.sum(axis=1) == c.TOTAL_SOLDIERS).all()
    assert lib.prior.sum() == pytest.approx(1.0)


def test_payoffs_are_constant_sum():
    rng = np.random.default_rng(1)
    a, b = c.level0_uniform(1, rng)[0], c.level0_uniform(50, rng)
    there = bf.payoff_vs(a, b)
    back = np.array([bf.payoff_vs(row, a[None])[0] for row in b])
    assert np.allclose(there + back, c.TOTAL_POINTS)


def test_field_scores_match_head_to_head_average():
    rng = np.random.default_rng(2)
    field = c.level0_uniform(40, rng)
    direct = np.array([bf.payoff_vs(a, field).mean() for a in field])
    assert np.allclose(bf.field_scores(field, np.ones(len(field))), direct)


def test_tilt_hits_every_anchor(lib):
    anchors = [(np.array([8, 17, 30, 44, 1]), 8.7), (np.array([0, 0, 0, 0, 100]), 5.6)]
    w = bf.tilt_to_anchors(lib.prior, lib.allocs, anchors, sweeps=20)
    wv = c.win_value(c.marginals(lib.allocs, w))
    for alloc, target in anchors:
        assert c.expected_score(wv, alloc[None])[0] == pytest.approx(target, abs=1e-3)


def test_e_step_conserves_board_count(lib):
    s = bf.field_scores(lib.allocs, lib.prior)
    y, cnt = np.array([6.0, 7.5, 9.0]), np.array([3, 5, 2])
    counts, _ = bf.e_step(y, cnt, s, lib.prior, sigma=0.05, chunk=97)
    assert counts.sum() == pytest.approx(cnt.sum())


def test_load_anchors_reads_current_scores(tmp_path):
    path = tmp_path / "anchors.csv"
    path.write_text("board_id,game,label,alloc,source\nx1,g1,on board,8-17-30-44-1,s\nx2,g1,gone,0-0-0-0-100,s\nx1,g2,other game,0-0-0-0-100,s\n")
    g1 = pd.DataFrame({"id": ["x1", "x3"], "p": [8.699, 7.0]})
    [(label, alloc, score)] = strategy.load_anchors(g1, path)
    assert label == "on board" and list(alloc) == [8, 17, 30, 44, 1] and score == 8.699

    path.write_text("board_id,game,label,alloc,source\nx1,g1,bad,8-17-30-44-2,s\n")
    with pytest.raises(ValueError):
        strategy.load_anchors(g1, path)


def test_anchor_resolves_by_full_name_once_on_board(tmp_path):
    path = tmp_path / "anchors.csv"
    path.write_text("board_id,game,label,alloc,source\nname:Ada Lovelace,g1,test entry,2-8-35-48-7,s\n")
    absent = pd.DataFrame({"id": ["x1"], "p": [8.0], "n": ["Someone Else"]})
    assert strategy.load_anchors(absent, path) == []
    present = pd.DataFrame({"id": ["x1", "x9"], "p": [8.0, 9.1], "n": ["Someone Else", "ada lovelace "]})
    [(label, alloc, score)] = strategy.load_anchors(present, path)
    assert score == 9.1
