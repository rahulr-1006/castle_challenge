import numpy as np
import pytest

from sim import castles as c
from live import robust as rb


def toy_member(name, allocs, weights, loo=(0.0,)):
    allocs = np.asarray(allocs)
    w = np.asarray(weights, dtype=float)
    br, _ = c.best_response(c.win_value(c.marginals(allocs, w / w.sum())))
    return rb.Member(name, allocs, w / w.sum(), np.asarray(loo), br, allocs[:2])


FIELD = [[20, 20, 20, 20, 20], [0, 0, 0, 0, 100], [10, 20, 30, 40, 0], [5, 7, 35, 47, 6]]


def test_neighbors_are_legal_one_soldier_moves():
    nb = rb.neighbors(np.array([20, 20, 20, 20, 20]))
    assert len(nb) == 20
    assert (nb.sum(axis=1) == 100).all()
    assert (np.abs(nb - 20).sum(axis=1) == 2).all()
    # A castle with no soldiers can't give one up.
    assert (rb.neighbors(np.array([0, 0, 0, 0, 100])) >= 0).all()


def test_member_weights_favor_better_out_of_sample_fit():
    good = toy_member("good", FIELD, [1, 1, 1, 1], loo=[0.05, -0.05])
    bad = toy_member("bad", FIELD, [1, 1, 1, 1], loo=[0.8, -0.6])
    w = rb.member_weights([good, bad])
    assert w[0] > w[1] and w.sum() == pytest.approx(1.0)


def test_late_entrants_have_requested_count_and_are_legal():
    m = toy_member("a", FIELD, [1, 1, 1, 1])
    late = rb._late_entrants(m, m, 1000, 250, 0.4, np.array([0.5, 0.3, 0.2]), 60.0,
                             np.random.default_rng(0))
    assert len(late) == 250
    assert (late.sum(axis=1) == 100).all() and (late >= 0).all()


def test_dominant_candidate_wins_every_world():
    m = toy_member("a", FIELD, [1, 1, 1, 1])
    wv = c.win_value(c.marginals(m.allocs, m.weights))
    strong, weak = c.best_response(wv)[0], np.array([0, 0, 0, 0, 100])
    s_strong = round(c.expected_score(wv, strong[None])[0], 3)
    worlds = rb.Worlds(np.zeros(3, dtype=int), np.stack([wv] * 3), np.full(3, s_strong - 0.5),
                       [np.array([s_strong - 0.5])] * 3, np.tile([200, 0.3, 0.5, 0.3, 0.2, 60.0], (3, 1)))
    res = rb.evaluate(np.stack([strong, weak]), worlds)
    assert res.p_first[0] == 1.0 and res.p_first[1] == 0.0
    assert rb.rank_candidates(res)[0] == 0
