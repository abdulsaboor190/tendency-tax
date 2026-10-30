"""Tests for the verdict function (Deliverable D4.1)."""


from src.estimate import BANNED_WORDS, VerdictResult, verdict


def _v(delta=0.05, ci_lo=0.01, ci_hi=0.09, early=0.03, p_lo=-0.02, p_hi=0.02):
    """Helper shortcut with fully passing defaults."""
    return verdict(delta, ci_lo, ci_hi, early, p_lo, p_hi)


# ---- All 8 combinations of the three conditions ----


def test_all_true_is_supported():
    """c1 AND c2 AND c3 → 'supported'."""
    res = _v(delta=0.05, ci_lo=0.01, ci_hi=0.09, early=0.03, p_lo=-0.02, p_hi=0.02)
    assert res.state == "supported"
    assert res.conditions["c1_ci_excludes_zero"] is True
    assert res.conditions["c2_sign_matches_early"] is True
    assert res.conditions["c3_outside_placebo"] is True
    assert res.failed == []
    assert isinstance(res, VerdictResult)


def test_c1_false_only():
    """CI straddles 0 → not supported; c1 in failed."""
    res = verdict(0.05, -0.01, 0.09, 0.03, -0.20, -0.10)
    assert res.state == "not supported / inconclusive"
    assert res.conditions["c1_ci_excludes_zero"] is False
    assert "c1_ci_excludes_zero" in res.failed


def test_c2_false_only():
    """Sign mismatch → not supported; c2 in failed."""
    res = verdict(0.05, 0.01, 0.09, -0.03, -0.20, -0.10)
    assert res.state == "not supported / inconclusive"
    assert res.conditions["c2_sign_matches_early"] is False
    assert "c2_sign_matches_early" in res.failed


def test_c3_false_only():
    """Inside placebo range → not supported; c3 in failed."""
    res = verdict(0.05, 0.01, 0.09, 0.03, -0.10, 0.10)
    assert res.state == "not supported / inconclusive"
    assert res.conditions["c3_outside_placebo"] is False
    assert "c3_outside_placebo" in res.failed


def test_c1_c2_false():
    """c1=F, c2=F, c3=T → not supported; both in failed."""
    res = verdict(0.05, -0.01, 0.09, -0.03, -0.20, -0.10)
    assert res.state == "not supported / inconclusive"
    assert "c1_ci_excludes_zero" in res.failed
    assert "c2_sign_matches_early" in res.failed
    assert "c3_outside_placebo" not in res.failed


def test_c1_c3_false():
    """c1=F, c2=T, c3=F → not supported."""
    res = verdict(0.05, -0.01, 0.09, 0.03, -0.10, 0.10)
    assert res.state == "not supported / inconclusive"
    assert "c1_ci_excludes_zero" in res.failed
    assert "c3_outside_placebo" in res.failed
    assert "c2_sign_matches_early" not in res.failed


def test_c2_c3_false():
    """c1=T, c2=F, c3=F → not supported."""
    res = verdict(0.05, 0.01, 0.09, -0.03, -0.10, 0.10)
    assert res.state == "not supported / inconclusive"
    assert "c2_sign_matches_early" in res.failed
    assert "c3_outside_placebo" in res.failed
    assert "c1_ci_excludes_zero" not in res.failed


def test_all_false():
    """All three conditions fail → not supported; all three in failed."""
    res = verdict(0.05, -0.01, 0.09, -0.03, -0.10, 0.10)
    assert res.state == "not supported / inconclusive"
    assert set(res.failed) == {
        "c1_ci_excludes_zero",
        "c2_sign_matches_early",
        "c3_outside_placebo",
    }


# ---- Boundary cases ----


def test_ci_endpoint_exactly_zero_fails_c1():
    """CI lo == 0.0 → c1 fails (strict inequality)."""
    res = verdict(0.05, 0.0, 0.09, 0.03, -0.10, -0.01)
    assert res.conditions["c1_ci_excludes_zero"] is False

    # CI hi == 0.0 also fails
    res2 = verdict(-0.05, -0.09, 0.0, -0.03, 0.01, 0.10)
    assert res2.conditions["c1_ci_excludes_zero"] is False


def test_delta_exactly_equal_to_placebo_bound_fails_c3():
    """delta exactly equals placebo_hi → c3 fails (strict inequality)."""
    res = verdict(0.05, 0.01, 0.09, 0.03, -0.02, 0.05)
    assert res.conditions["c3_outside_placebo"] is False

    # delta exactly equals placebo_lo
    res2 = verdict(-0.05, -0.09, -0.01, -0.03, -0.05, 0.02)
    assert res2.conditions["c3_outside_placebo"] is False


def test_early_delta_exactly_zero_fails_c2():
    """early_delta == 0.0 → c2 fails."""
    res = verdict(0.05, 0.01, 0.09, 0.0, -0.02, -0.01)
    assert res.conditions["c2_sign_matches_early"] is False


def test_holdout_delta_exactly_zero_fails_c2():
    """holdout_delta == 0.0 → c2 fails."""
    res = verdict(0.0, -0.01, 0.01, 0.05, -0.10, -0.05)
    assert res.conditions["c2_sign_matches_early"] is False


# ---- Direction and wording ----


def test_positive_direction():
    res = _v(delta=0.05)
    assert res.direction == "positive"


def test_negative_direction():
    res = verdict(-0.05, -0.09, -0.01, -0.03, -0.20, -0.10)
    assert res.direction == "negative"
    assert res.state == "supported"
    assert "in line with the situational expectation" in res.sentence


def test_zero_direction():
    res = verdict(0.0, -0.01, 0.01, 0.05, 0.10, 0.20)
    assert res.direction == "zero"


def test_positive_supported_wording():
    res = _v()
    assert res.state == "supported"
    assert "associated with higher EPA" in res.sentence
    assert "cost of predictability" in res.sentence


def test_not_supported_wording_mentions_failures():
    res = verdict(0.05, -0.01, 0.09, -0.03, -0.10, 0.10)
    assert "no reliable evidence" in res.sentence
    assert "confidence interval" in res.sentence or "sign" in res.sentence


# ---- Banned-words test ----


def test_no_banned_words_in_sentence():
    """No sentence contains causal language."""
    test_cases = [
        _v(),  # supported positive
        verdict(-0.05, -0.09, -0.01, -0.03, -0.20, -0.10),  # supported negative
        verdict(0.05, -0.01, 0.09, -0.03, -0.10, 0.10),  # all failed
        verdict(0.05, 0.0, 0.09, 0.03, -0.10, -0.01),  # c1 boundary
    ]
    for res in test_cases:
        words_in_sentence = set(res.sentence.lower().split())
        for banned in BANNED_WORDS:
            assert banned not in words_in_sentence, (
                f"Banned word '{banned}' found in sentence: {res.sentence!r}"
            )


# ---- to_dict ----


def test_to_dict_returns_dict():
    res = _v()
    d = res.to_dict()
    assert isinstance(d, dict)
    assert "state" in d
    assert "conditions" in d
    assert "failed" in d
    assert "direction" in d
    assert "sentence" in d
