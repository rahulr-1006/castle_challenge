import itertools
import math
from fractions import Fraction

import numpy as np
import pytest

from sim import castles as c


# ---------------------------------------------------------------------------
# Slow, obviously-correct reference implementations of the written rules.
# The vectorized code in castles.py is checked against these on random inputs.
# ---------------------------------------------------------------------------

def ref_castle_points(group):
    pts = [Fraction(0)] * len(group)
    for castle, value in enumerate(c.VALUES):
        sent = [alloc[castle] for alloc in group]
        winners = [i for i, s in enumerate(sent) if s == max(sent)]
        for i in winners:
            pts[i] += Fraction(int(value), len(winners))
    return pts


def ref_rank_payouts(points):
    order = sorted(points, reverse=True)
    out = []
    for p in points:
        places = [i for i, q in enumerate(order) if q == p]
        out.append(sum(c.PAYOUTS[i] for i in places) / len(places))
    return out


def random_allocs(rng, n, max_cluster=None):
    """Random legal allocations; `max_cluster` coarsens them so ties are common."""
    if max_cluster is None:
        cuts = np.sort(rng.integers(0, c.TOTAL_SOLDIERS + 1, size=(n, c.N_CASTLES - 1)), axis=1)
        edges = np.concatenate([np.zeros((n, 1), int), cuts, np.full((n, 1), c.TOTAL_SOLDIERS)], axis=1)
        return np.diff(edges, axis=1)
    # Multiples of 100 / max_cluster, so equal castle counts show up constantly.
    units = random_allocs(rng, n) * max_cluster // c.TOTAL_SOLDIERS
    units[:, -1] += max_cluster - units.sum(axis=1)
    return units * (c.TOTAL_SOLDIERS // max_cluster)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [
    [20, 20, 20, 20, 19],          # sums to 99
    [20, 20, 20, 20, 21],          # sums to 101
    [-1, 21, 20, 20, 40],          # negative
    [20.5, 19.5, 20, 20, 20],      # non-integer
    [25, 25, 25, 25],              # wrong length
])
def test_validate_rejects_illegal(bad):
    with pytest.raises(ValueError):
        c.validate(bad)


def test_validate_accepts_integral_floats_and_batches():
    out = c.validate([[20.0, 20, 20, 20, 20], [0, 0, 0, 0, 100]])
    assert out.dtype == np.int64 and out.shape == (2, 5)


# ---------------------------------------------------------------------------
# Game 1
# ---------------------------------------------------------------------------

def test_game1_sweep_and_mirror():
    a = [10, 15, 20, 25, 30]
    b = [9, 14, 19, 24, 34]  # a wins castles 1-4, loses castle 5
    assert c.game1_points(a, b)[0] == 10
    assert c.game1_points(a, a)[0] == 7.5


def test_game1_castle_tie_splits_value():
    a = [20, 20, 20, 20, 20]
    b = [20, 10, 30, 40, 0]  # tie on 1, win 2, lose 3, lose 4, win 5
    assert c.game1_points(a, b)[0] == 0.5 + 2 + 5


def test_game1_is_constant_sum():
    rng = np.random.default_rng(0)
    allocs = random_allocs(rng, 400, max_cluster=10)
    for a, b in zip(allocs[::2], allocs[1::2]):
        assert c.game1_points(a, b)[0] + c.game1_points(b, a)[0] == c.TOTAL_POINTS


def test_game1_score_averages_over_field():
    me = [20, 20, 20, 20, 20]
    field = [[20, 20, 20, 20, 20], [0, 0, 0, 0, 100], [100, 0, 0, 0, 0]]
    assert c.game1_score(me, field) == pytest.approx((7.5 + 10 + 14) / 3)


# ---------------------------------------------------------------------------
# Game 2: castle ties
# ---------------------------------------------------------------------------

def points(group):
    return c.game2_castle_points(np.array(group)) / c.SCALE


def test_game2_two_way_castle_tie():
    group = [[0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [50, 50, 0, 0, 0],
             [0, 0, 50, 50, 0]]
    # Castle 5 splits 2.5 / 2.5. Castles 1, 2 to player 2; castles 3, 4 to player 3.
    np.testing.assert_allclose(points(group), [2.5, 2.5, 3, 7])


def test_game2_three_way_castle_tie_is_exact():
    group = [[0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [0, 0, 0, 0, 100],
             [10, 20, 30, 40, 0]]
    pts = c.game2_castle_points(np.array(group))
    # 5/3 each: equal in integer twelfths, so the ranking sees an exact 3-way tie.
    assert pts[0] == pts[1] == pts[2] == 20
    assert pts[3] == 10 * c.SCALE
    np.testing.assert_allclose(c.rank_payouts(pts), [2, 2, 2, 9])


def test_game2_four_way_castle_tie_including_zeros():
    group = [[20, 20, 20, 20, 20]] * 4
    np.testing.assert_allclose(points(group), [3.75] * 4)
    # Nobody sends anything to castles 1-2: those still split four ways at zero.
    group = [[0, 0, 100, 0, 0], [0, 0, 0, 100, 0], [0, 0, 0, 0, 100], [0, 0, 0, 0, 100]]
    np.testing.assert_allclose(points(group), [0.75 + 3, 0.75 + 4, 0.75 + 2.5, 0.75 + 2.5])


def test_game2_castle_points_always_total_15():
    rng = np.random.default_rng(1)
    groups = random_allocs(rng, 4000, max_cluster=10).reshape(-1, 4, 5)
    assert (c.game2_castle_points(groups).sum(axis=-1) == c.TOTAL_POINTS * c.SCALE).all()


# ---------------------------------------------------------------------------
# Game 2: ranking ties
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pts, expected", [
    ([4, 3, 2, 1], [9, 4, 2, 0]),        # strict order
    ([1, 4, 2, 3], [0, 9, 2, 4]),        # order independent of seat
    ([5, 5, 3, 2], [6.5, 6.5, 2, 0]),    # 2-way tie for 1st
    ([6, 4, 4, 1], [9, 3, 3, 0]),        # 2-way tie for 2nd
    ([6, 5, 2, 2], [9, 4, 1, 1]),        # 2-way tie for 3rd
    ([5, 5, 2, 2], [6.5, 6.5, 1, 1]),    # two pairs
    ([5, 5, 5, 0], [5, 5, 5, 0]),        # 3-way tie for 1st
    ([9, 2, 2, 2], [9, 2, 2, 2]),        # 3-way tie for 2nd
    ([3, 3, 3, 3], [3.75] * 4),          # 4-way tie
])
def test_rank_payouts_ties(pts, expected):
    np.testing.assert_allclose(c.rank_payouts(np.array(pts)), expected)


def test_payouts_always_total_15():
    rng = np.random.default_rng(2)
    groups = random_allocs(rng, 4000, max_cluster=5).reshape(-1, 4, 5)
    np.testing.assert_allclose(c.game2_payouts(groups).sum(axis=-1), c.PAYOUTS.sum())


# ---------------------------------------------------------------------------
# Vectorized rules == reference rules on random groups
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_cluster", [None, 20, 10, 5, 4])
def test_game2_matches_reference(max_cluster):
    rng = np.random.default_rng(3)
    groups = random_allocs(rng, 2000, max_cluster).reshape(-1, 4, 5)
    pts = c.game2_castle_points(groups)
    pay = c.game2_payouts(groups)
    for g, p, y in zip(groups, pts, pay):
        ref = ref_castle_points(g.tolist())
        assert [Fraction(int(x), c.SCALE) for x in p] == ref
        np.testing.assert_allclose(y, ref_rank_payouts(ref))


def test_game1_matches_reference():
    rng = np.random.default_rng(4)
    me = random_allocs(rng, 1, max_cluster=10)[0]
    field = random_allocs(rng, 500, max_cluster=10)
    got = c.game1_points(me, field)
    for opp, score in zip(field, got):
        # Game 1 is a 2-player Game 2 castle split; reuse the reference.
        assert score == float(ref_castle_points([me.tolist(), opp.tolist()])[0])


# ---------------------------------------------------------------------------
# Allocation space and rounding
# ---------------------------------------------------------------------------

def test_all_allocations_is_every_legal_allocation_once():
    allocs = c.all_allocations()
    assert len(allocs) == math.comb(104, 4)
    c.validate(allocs)
    assert len(np.unique(allocs.astype(np.int64) @ (101 ** np.arange(5)))) == len(allocs)


def test_round_to_total():
    rng = np.random.default_rng(5)
    shares = rng.dirichlet(np.ones(5), size=1000)
    for step in (1, 5, 10):
        out = c.round_to_total(shares, step)
        assert (out.sum(axis=1) == 100).all() and (out % step == 0).all()
        assert np.abs(out - 100 * shares).max() <= step
    out = c.round_to_total(shares[:3], 5, total=np.array([40, 50, 60]))
    assert out.sum(axis=1).tolist() == [40, 50, 60]


# ---------------------------------------------------------------------------
# Fact 1: Game 1 is separable, and the DP best response is exact
# ---------------------------------------------------------------------------

def small_board(n_castles, soldiers):
    return [a for a in itertools.product(range(soldiers + 1), repeat=n_castles) if sum(a) == soldiers]


def test_separable_score_matches_rules():
    rng = np.random.default_rng(6)
    field = random_allocs(rng, 300, max_cluster=10)
    w = c.win_value(c.marginals(field))
    for a in random_allocs(rng, 50, max_cluster=20):
        assert c.expected_score(w, a) == pytest.approx(c.game1_score(a, field))


@pytest.mark.parametrize("n_castles, soldiers", [(2, 7), (3, 9), (4, 12), (5, 10)])
def test_best_response_matches_brute_force_on_small_boards(n_castles, soldiers):
    rng = np.random.default_rng(n_castles * 100 + soldiers)
    board = np.array(small_board(n_castles, soldiers))
    for _ in range(50):
        # Integer payoffs make exact ties common, so the value comparison is strict.
        w = rng.integers(0, 6, size=(n_castles, soldiers + 1)).astype(float)
        alloc, value = c.best_response(w)
        assert alloc.sum() == soldiers and (alloc >= 0).all()
        assert value == c.expected_score(w, alloc)
        assert value == c.expected_score(w, board).max()


def test_best_response_matches_exhaustive_search_on_real_board():
    rng = np.random.default_rng(7)
    allocs = c.all_allocations()
    fields = [c.level0_sample("focal", 5000, rng), random_allocs(rng, 2000), random_allocs(rng, 2000, 10)]
    for field in fields:
        w = c.win_value(c.marginals(field))
        _, value = c.best_response(w)
        assert value == pytest.approx(c.expected_score(w, allocs).max(), abs=1e-12)


# ---------------------------------------------------------------------------
# Noisy best response (logit over all allocations)
# ---------------------------------------------------------------------------

def test_logit_marginals_match_enumeration():
    rng = np.random.default_rng(8)
    board = np.array(small_board(4, 12))
    for lam in (0.0, 0.7, 3.0):
        w = rng.random((4, 13)) * 3
        p = np.exp(lam * c.expected_score(w, board))
        p /= p.sum()
        exact = np.stack([np.bincount(board[:, i], weights=p, minlength=13) for i in range(4)])
        np.testing.assert_allclose(c.logit_marginals(w, lam), exact, atol=1e-12)


def test_logit_with_zero_lambda_is_uniform_on_real_board():
    uniform = np.array([math.comb(100 - s + 3, 3) for s in range(101)]) / math.comb(104, 4)
    np.testing.assert_allclose(c.logit_marginals(np.zeros((5, 101)), 0.0), np.tile(uniform, (5, 1)), atol=1e-15)


def test_logit_sample_matches_exact_probabilities():
    rng = np.random.default_rng(9)
    board = np.array(small_board(4, 8))
    w = rng.random((4, 9)) * 3
    p = np.exp(1.5 * c.expected_score(w, board))
    p /= p.sum()
    draws = c.logit_sample(w, 1.5, 200_000, rng)
    assert (draws.sum(axis=1) == 8).all() and (draws >= 0).all()
    index = {tuple(a): i for i, a in enumerate(board.tolist())}
    freq = np.bincount([index[tuple(d)] for d in draws.tolist()], minlength=len(board)) / len(draws)
    # Each cell's binomial standard error is at most ~0.0011 here.
    assert np.abs(freq - p).max() < 0.006


def test_logit_sample_on_real_board_and_large_lambda():
    rng = np.random.default_rng(10)
    w = c.win_value(c.marginals(c.level0_sample("focal", 20_000, rng)))
    draws = c.validate(c.logit_sample(w, 5.0, 100_000, rng))
    assert np.abs(c.marginals(draws) - c.logit_marginals(w, 5.0)).max() < 0.006
    # As lam grows the noisy response closes in on the exact best response.
    _, best = c.best_response(w)
    gaps = [best - (c.logit_marginals(w, lam) * w).sum() for lam in (2, 8, 40)]
    assert gaps[0] > gaps[1] > gaps[2] > 0 and gaps[2] < 0.1


# ---------------------------------------------------------------------------
# Fact 2: fictitious play converges toward Game 1's mixed equilibrium
# ---------------------------------------------------------------------------

def test_fictitious_play_converges():
    fp = c.fictitious_play(5000)
    gaps = fp.exploitability
    assert gaps[99] > gaps[999] > gaps[4999]
    assert gaps[4999] < 0.25
    # The running average is the plain average of every play so far, and each
    # reported gap is what a best response gains against the average at that round.
    np.testing.assert_allclose(fp.marginals, c.marginals(fp.plays), atol=1e-12)
    for t in (0, 10, 500):
        _, value = c.best_response(c.win_value(c.marginals(fp.plays[: t + 1])))
        assert gaps[t] == pytest.approx(value - 7.5)


def test_fictitious_play_matches_gross_wagner():
    # Continuous Blotto equilibrium (Gross and Wagner 1950): castle i's soldiers
    # are uniform on [0, 2 * 100 * v_i / 15]. The integer game should be close.
    fp = c.fictitious_play(20_000)
    for i, v in enumerate(c.VALUES):
        top = 200 * v / 15
        gross_wagner = np.clip((np.arange(101) + 0.5) / top, 0, 1)
        assert np.abs(np.cumsum(fp.marginals[i]) - gross_wagner).max() < 0.08
    np.testing.assert_allclose(c.mean_allocation(fp.marginals), c.PROPORTIONAL, atol=1)


# ---------------------------------------------------------------------------
# Player types
# ---------------------------------------------------------------------------

def test_level0_habits_are_legal():
    rng = np.random.default_rng(11)
    for draw in c.LEVEL0_HABITS.values():
        c.validate(draw(2000, rng))
    assert (c.level0_round_numbers(2000, rng) % 5 == 0).all()
    assert set(c.level0_stack_five(2000, rng)[:, 4]) == {40, 50, 60}
    for flavor in c.LEVEL0_FLAVORS:
        assert c.validate(c.level0_sample(flavor, 3000, rng)).shape == (3000, 5)


def test_poisson_levels():
    shares = c.poisson_levels(1.5, 6)
    assert shares.sum() == pytest.approx(1)
    assert shares[2] == pytest.approx(math.exp(-1.5) * 1.5**2 / 2)
    assert shares[6] > math.exp(-1.5) * 1.5**6 / math.factorial(6)  # pooled tail


def test_cognitive_hierarchy_levels():
    rng = np.random.default_rng(12)
    level0 = c.marginals(c.level0_sample("focal", 50_000, rng))
    exact = c.cognitive_hierarchy(level0, tau=1.5, lam=np.inf, max_level=3)
    noisy = c.cognitive_hierarchy(level0, tau=1.5, lam=3.0, max_level=3)
    assert exact.shape == noisy.shape == (4, 5, 101)
    np.testing.assert_allclose(exact[0], level0)
    # Exact level 1 is a point mass on the best response to level 0.
    np.testing.assert_allclose(exact[1], c.marginals(c.best_response(c.win_value(level0))[0]))
    # Level 2 best-responds to levels 0 and 1 weighted by their Poisson shares.
    shares = c.poisson_levels(1.5, 3)[:2]
    belief = (shares[0] * level0 + shares[1] * exact[1]) / shares.sum()
    np.testing.assert_allclose(exact[2], c.marginals(c.best_response(c.win_value(belief))[0]))
    np.testing.assert_allclose(noisy.sum(axis=2), 1)


# ---------------------------------------------------------------------------
# Game 2 simulator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_cluster", [None, 10, 5])
def test_payouts_match_reference_rules(max_cluster):
    rng = np.random.default_rng(13)
    opps = random_allocs(rng, 3 * 1500, max_cluster).reshape(-1, 3, 5)
    cands = random_allocs(rng, 12, max_cluster)
    got = c.payouts(cands, c.face(opps), max_cells=5000)  # small chunks exercise chunking
    for a, row in zip(cands, got):
        groups = np.concatenate([np.broadcast_to(a, (len(opps), 1, 5)), opps], axis=1)
        np.testing.assert_allclose(row, c.game2_payouts(groups)[:, 0])


def test_more_than_half_the_points_always_wins_outright():
    rng = np.random.default_rng(14)
    groups = random_allocs(rng, 4 * 20_000, max_cluster=10).reshape(-1, 4, 5)
    pts = c.game2_castle_points(groups)
    pay = c.game2_payouts(groups)
    over_half = pts * 2 > c.TOTAL_POINTS * c.SCALE
    assert over_half.sum() > 1000
    assert (pay[over_half] == 9).all()


def test_exactly_half_the_points_can_still_tie():
    group = np.array([[10, 0, 45, 45, 0],     # ties castle 1, wins 3 and 4: 7.5
                      [10, 30, 0, 0, 60],     # ties castle 1, wins 2 and 5: 7.5
                      [0, 25, 25, 25, 25],
                      [0, 25, 25, 25, 25]])
    np.testing.assert_allclose(c.game2_castle_points(group) / c.SCALE, [7.5, 7.5, 0, 0])
    np.testing.assert_allclose(c.game2_payouts(group), [6.5, 6.5, 1, 1])


def test_minimal_eight_point_sets():
    sets = [s for r in range(1, 6) for s in itertools.combinations(range(1, 6), r) if sum(s) >= 8]
    minimal = {s for s in sets if not any(set(t) < set(s) for t in sets)}
    assert minimal == {(3, 5), (4, 5), (1, 2, 5), (1, 3, 4), (2, 3, 4)}


def test_castle_share_4way_matches_enumeration():
    rng = np.random.default_rng(15)
    allocs = random_allocs(rng, 6, max_cluster=10)
    probs = rng.dirichlet(np.ones(6))
    q = c.castle_share_4way(c.marginals(allocs, probs))
    for castle in range(5):
        for s in range(0, 101, 5):
            exact = 0.0
            for trio in itertools.product(range(6), repeat=3):
                sent = allocs[list(trio), castle]
                if s >= sent.max():
                    exact += np.prod(probs[list(trio)]) / (1 + (sent == s).sum())
            assert q[castle, s] == pytest.approx(exact)


def test_game2_response_wins_everything_against_equal_split():
    rng = np.random.default_rng(16)
    r = c.game2_response(c.field_from_samples([c.EQUAL_SPLIT]), rng, n_groups=200)
    assert r.values.max() == 9


def test_local_search_and_annealing_climb():
    rng = np.random.default_rng(17)
    field = c.field_from_samples(c.level0_sample("focal", 20_000, rng))
    opp = c.face(c.sample_opponents(field, 2000, rng))
    start_value = c.payouts(c.EQUAL_SPLIT, opp).mean()
    best, value = c.local_search(c.EQUAL_SPLIT, opp)
    assert value > start_value + 1
    neighbors = best + c._MOVES
    neighbors = neighbors[(neighbors >= 0).all(axis=1)]
    assert c.payouts(neighbors, opp).mean(axis=1).max() <= value  # a true local optimum
    annealed, annealed_value = c.anneal(c.EQUAL_SPLIT, opp, 400, rng)
    assert annealed_value == pytest.approx(c.payouts(annealed, opp).mean())
    assert annealed_value > start_value


def test_logit_field_and_mixing():
    r = c.Game2Response(np.array([[20, 20, 20, 20, 20], [0, 0, 40, 0, 60]]), np.array([3.0, 5.0]))
    assert c.logit_field(r, np.inf).allocs.tolist() == [[0, 0, 40, 0, 60]]
    np.testing.assert_allclose(c.logit_field(r, 0.0).probs, [0.5, 0.5])
    np.testing.assert_allclose(c.logit_field(r, 1.0).probs, np.exp([-2, 0]) / np.exp([-2, 0]).sum())
    mixed = c.mix_fields([(3, c.logit_field(r, 0.0)), (1, c.field_from_samples([c.EQUAL_SPLIT])), (0, r)])
    np.testing.assert_allclose(mixed.probs, [0.375, 0.375, 0.25])


def test_replicator_drives_out_a_dominated_plan():
    rng = np.random.default_rng(18)
    pool = np.array([c.EQUAL_SPLIT, [16, 21, 21, 21, 21]])
    mix = c.replicator(pool, rng, steps=100, n_groups=500)
    assert mix.probs[1] > 0.9


def test_game2_fictitious_play_reduces_exploitability():
    rng = np.random.default_rng(19)
    plays, gaps = c.fictitious_play_game2(150, rng, n_groups=1000, checkpoints=(10, 150))
    assert len(plays) == 151
    c.validate(plays)
    assert gaps[150] < gaps[10] - 1
