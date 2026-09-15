"""Tier 1: regression.py's own logic, a disposable tmp_path directory,
no network, no model. See chapter 29.
"""

from reliable_agents_labs.regression import check_for_regression, load_baseline, save_baseline


def test_no_baseline_yet_is_not_a_regression(tmp_path):
    assert check_for_regression("demo", 0.5, directory=tmp_path) is None


def test_a_real_drop_past_tolerance_is_flagged(tmp_path):
    save_baseline("demo", 1.0, directory=tmp_path)

    message = check_for_regression("demo", 0.6, tolerance=0.1, directory=tmp_path)

    assert message is not None
    assert "1.00" in message
    assert "0.60" in message


def test_a_small_fluctuation_within_tolerance_is_not_flagged(tmp_path):
    save_baseline("demo", 1.0, directory=tmp_path)

    assert check_for_regression("demo", 0.92, tolerance=0.1, directory=tmp_path) is None


def test_save_and_load_round_trip(tmp_path):
    save_baseline("demo", 0.875, directory=tmp_path)

    assert load_baseline("demo", directory=tmp_path).pass_rate == 0.875
