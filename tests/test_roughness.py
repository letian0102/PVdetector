import numpy as np

from peak_valley.roughness import (
    density_metric,
    find_bw_for_roughness,
    has_small_double_peak,
    _compute_density,
)


def test_density_metric_reports_extrema_and_roughness():
    xs = np.linspace(-1.0, 1.0, 9)
    ys = 1.0 - xs**2  # parabola with clear maximum at x=0

    metric = density_metric(xs, ys, tol=0.05)

    assert metric.n_x == xs.size
    assert metric.roughness >= 0
    assert metric.zero_idx.size > 0
    assert metric.big_idx.size > 0


def test_roughness_bandwidth_prefers_smoother_profile():
    rng = np.random.default_rng(42)
    data = np.concatenate(
        [rng.normal(0.0, 0.05, 600), rng.normal(0.2, 0.05, 200)]
    )

    lower = 0.01
    upper = 0.2
    selected_bw = find_bw_for_roughness(
        data, lower=lower, upper=upper, grid_size=800
    )

    xs_low, ys_low = _compute_density(data, lower, 800)
    xs_sel, ys_sel = _compute_density(data, selected_bw, 800)

    low_metric = density_metric(xs_low, ys_low)
    sel_metric = density_metric(xs_sel, ys_sel)

    assert lower <= selected_bw <= upper
    assert sel_metric.roughness <= low_metric.roughness
    assert not has_small_double_peak(xs_sel, ys_sel)
