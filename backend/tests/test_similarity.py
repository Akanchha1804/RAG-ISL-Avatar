from similarity import distance_to_similarity, exact_match_similarity


def test_distance_zero_is_one():
    assert distance_to_similarity(0.0) == 1.0


def test_known_values():
    assert distance_to_similarity(1.0) == 0.5
    assert abs(distance_to_similarity(3.0) - 0.25) < 1e-9


def test_monotonic_decreasing():
    values = [distance_to_similarity(d) for d in (0.0, 0.5, 1.0, 2.0)]
    assert values == sorted(values, reverse=True)


def test_exact_match_similarity_is_one():
    assert exact_match_similarity() == 1.0


def test_no_overlap_distance_matches_fake_contract():
    # mirrors conftest fake: dist 2.0 -> sim 1/3 < match threshold 0.4
    assert abs(distance_to_similarity(2.0) - (1.0 / 3.0)) < 1e-9
