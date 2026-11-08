import numpy as np
import pytest

from live import board_fit as bf
from sim import castles as c
from live import newcomers as nc


def snapshot(field, ids, rounded=False):
    """Board as the site would publish it: each entry's mean payoff against all others."""
    entries = []
    for i, (a, id_) in enumerate(zip(field, ids)):
        others = np.delete(field, i, axis=0)
        p = bf.payoff_vs(a, others).mean()
        entries.append({"id": id_, "p": round(p, 3) if rounded else p})
    return {"g1": entries}


def test_cohort_reads_recover_average_payoff_against_newcomers():
    rng = np.random.default_rng(0)
    old = c.level0_uniform(60, rng)
    new = c.level0_uniform(15, rng)
    ids = [f"e{i}" for i in range(75)]
    first = snapshot(old, ids[:60])
    last = snapshot(np.concatenate([old, new]), ids)
    anchors = [("a", ids[0], old[0]), ("b", ids[7], old[7]), ("gone", "nope", old[1])]
    reads = nc.cohort_reads(anchors, first, last)
    assert [r.label for r in reads] == ["a", "b"]
    for r, i in zip(reads, (0, 7)):
        assert r.k == 15 and r.removed == 0
        assert r.mean_payoff == pytest.approx(bf.payoff_vs(old[i], new).mean(), abs=1e-9)


def test_rounded_reads_stay_within_reported_noise():
    rng = np.random.default_rng(1)
    old = c.level0_uniform(400, rng)
    new = c.level0_uniform(40, rng)
    ids = [f"e{i}" for i in range(440)]
    first = snapshot(old, ids[:400], rounded=True)
    last = snapshot(np.concatenate([old, new]), ids, rounded=True)
    anchors = [(f"a{i}", ids[i], old[i]) for i in range(30)]
    errors = [r.mean_payoff - bf.payoff_vs(r.alloc, new).mean()
              for r in nc.cohort_reads(anchors, first, last)]
    rounding_sd = np.hypot(399, 439) * nc.ROUND_SD / 40
    assert np.max(np.abs(errors)) < 5 * rounding_sd


def test_soft_tilt_moves_toward_reads_without_forcing_them():
    lib = bf.build_library(np.random.default_rng(2), n=300)
    alloc = np.array([0, 0, 0, 0, 100])
    prior_pred = lib.prior @ bf.payoff_vs(alloc, lib.allocs)
    read = nc.CohortRead("x", alloc, prior_pred + 1.0, 0.5, 30, 0)
    w = nc.soft_tilt(lib.prior, lib.allocs, [read])
    pred = w @ bf.payoff_vs(alloc, lib.allocs)
    assert prior_pred < pred < prior_pred + 1.0
    assert w.sum() == pytest.approx(1.0)


def test_late_prior_needs_enough_newcomers():
    base = {"sharp_share": (0.2, 0.6), "level_mix": (1.0, 1.0, 1.0)}
    shares = {"plain": 0.6, "pasted": 0.1, "L1": 0.2, "L2": 0.05, "L3": 0.05}
    small = nc.Cohort([], None, shares, {}, {}, nc.MIN_COHORT - 1)
    assert nc.late_prior(small, base) is base
    big = small._replace(k=nc.MIN_COHORT)
    late = nc.late_prior(big, base)
    assert late["sharp_share"] == pytest.approx((0.3, 0.55))
    assert late["level_mix"][0] > late["level_mix"][1]


def test_known_newcomers_are_subtracted_not_counted():
    rng = np.random.default_rng(3)
    old = c.level0_uniform(60, rng)
    new = c.level0_uniform(15, rng)
    ids = [f"e{i}" for i in range(75)]
    first = snapshot(old, ids[:60])
    last = snapshot(np.concatenate([old, new]), ids)
    # The first two newcomers are test entries with known splits.
    anchors = [("a", ids[0], old[0]), ("f1", ids[60], new[0]), ("f2", ids[61], new[1])]
    [read] = nc.cohort_reads(anchors, first, last)
    assert read.k == 13
    assert read.mean_payoff == pytest.approx(bf.payoff_vs(old[0], new[2:]).mean(), abs=1e-9)
