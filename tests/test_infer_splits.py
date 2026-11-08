import numpy as np
import pytest

from live import infer_splits as inf
from live import robust as rb
from sim import castles as c

FIELD = np.array([[20, 20, 20, 20, 20], [0, 0, 0, 0, 100], [10, 20, 30, 40, 0],
                  [5, 7, 35, 47, 6], [4, 7, 35, 47, 7], [7, 13, 20, 27, 33]])


def member(weights, loo=(0.01, -0.01)):
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    br, _ = c.best_response(c.win_value(c.marginals(FIELD, w)))
    return rb.Member("toy", FIELD, w, np.asarray(loo), br, FIELD[:2])


@pytest.fixture(scope="module")
def tables():
    return inf.build_tables([member([3, 1, 2, 2, 1, 2])], [1.0])


def test_split_index_round_trips_and_rejects_illegal():
    rows = np.random.default_rng(0).integers(0, inf.N_ALL, 500)
    assert (inf.split_index(inf.ALL[rows]) == rows).all()
    with pytest.raises(ValueError):
        inf.split_index([50, 50, 50, 0, 0])


def test_window_matches_brute_force(tables):
    # Truncating at WINDOW sds may only drop mass far below anything that matters.
    s, eps, sigma = 8.5, 1e-3, 0.02
    rows, p = inf.posterior(tables, s, eps, sigma)
    order = np.argsort(tables.idx[0])
    pred = tables.score[0][order].astype(float)
    prior = (1 - eps) * tables.lib_w[0][order] + eps / inf.N_ALL
    var = sigma**2 + inf.ROUND_VAR
    logw = np.log(prior) - 0.5 * (s - pred) ** 2 / var
    full = np.exp(logw - logw.max())
    full /= full.sum()
    assert np.abs(full[rows] - p).max() < 1e-4
    assert full[rows].sum() > 0.999


def test_posterior_finds_a_common_split_from_its_exact_score(tables):
    truth = FIELD[0]
    s = float(c.expected_score(c.win_value(c.marginals(FIELD, member([3, 1, 2, 2, 1, 2]).weights)),
                               truth[None])[0])
    rows, p = inf.posterior(tables, s, eps=1e-4, sigma=0.002)
    assert inf.ALL[rows[np.argmax(p)]].tolist() == truth.tolist()
    assert abs(p.sum() - 1) < 1e-9


def test_score_truth_covers_the_grid_and_summary_brackets_the_map(tables):
    out = inf.score_truth(tables, 8.0, FIELD[2])
    assert len(out) == len(inf.EPS_GRID) * len(inf.SIGMA_GRID)
    assert all(0.0 <= r["p_true"] <= 1.0 for r in out)
    summ = inf.summarize(*inf.posterior(tables, 8.0, 1e-2, 0.05))
    map_split = [int(x) for x in summ["map"].split("-")]
    for k in range(5):
        lo, hi = (int(x) for x in summ[f"c{k + 1}_90"].split("-"))
        assert lo <= hi
    assert 0.0 <= summ["p_map"] <= summ["p_within_1"] <= 1.0 + 1e-12
    assert sum(map_split) == 100


def test_probe_walk_only_climbs_from_better_bases():
    rng = np.random.default_rng(1)
    field = FIELD[np.random.default_rng(2).integers(0, len(FIELD), 300)]
    probes = inf._probe_walk(FIELD[3], field, 8, rng)
    assert len(probes) == 8 and (probes.sum(axis=1) == 100).all()
    # Each probe is one soldier from the base it was tested against, so from some
    # earlier probe or the start.
    seen = [FIELD[3]] + list(probes)
    for k, p in enumerate(probes):
        assert any(np.abs(p - b).sum() == 2 for b in seen[:k + 1])


def test_ranks_match_site_convention():
    board = np.array([9.0, 8.5, 9.0, 7.0])
    assert inf._ranks(board).tolist() == [1, 3, 1, 4]
    assert inf.band_of(1) == "1-10" and inf.band_of(3500) == "3001-1000000"


def test_measured_split_counts_only_at_its_own_score(tables):
    split = FIELD[3]
    measured = inf.known_splits([(split, 9.227)])
    row = inf.split_index(split)[0]
    rows, p = inf.posterior(tables, 9.100, 1e-2, 0.06, measured)
    assert row not in set(rows[p > 0].tolist())  # measured at 9.227, so it can't be at 9.100
    rows, p = inf.posterior(tables, 9.227, 1e-2, 0.06, measured)
    assert rows[np.argmax(p)] == row  # an exact match beats any model-only candidate
