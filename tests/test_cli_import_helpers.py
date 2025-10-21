import math

from peak_valley.cli_import import derive_min_separation, parse_peak_positions


def test_parse_peak_positions_from_string():
    peaks = parse_peak_positions("1.0; 2.5;3.75")
    assert peaks == [1.0, 2.5, 3.75]


def test_parse_peak_positions_skips_bad_entries():
    peaks = parse_peak_positions(["1.0", "oops", 4.2, None])
    assert peaks == [1.0, 4.2]


def test_derive_min_separation_returns_margin_when_smaller_than_baseline():
    derived = derive_min_separation([0.0, 2.0, 5.0], baseline=6.0)
    assert 0.0 < derived < 2.0
    assert not math.isclose(derived, 6.0)


def test_derive_min_separation_none_when_spacing_meets_baseline():
    assert derive_min_separation([0.0, 8.0], baseline=6.0) is None


def test_derive_min_separation_handles_missing_or_duplicate_peaks():
    # Duplicate peak values should not produce negative gaps
    assert derive_min_separation([1.0, 1.0]) is None
