import numpy as np

from peak_valley import batch as batch_mod
from peak_valley.batch import BatchOptions
from peak_valley.kde_detector import kde_peaks_valleys


def test_ms_bandwidth_resolves_numeric_params():
    data = np.concatenate([
        np.random.normal(-3.0, 0.25, size=300),
        np.random.normal(3.0, 0.30, size=300),
    ])

    options = BatchOptions(apply_arcsinh=False, bandwidth="MS")
    params, debug = batch_mod._resolve_parameters(options, {}, None, data, None)

    assert isinstance(params["bandwidth_effective"], float)
    assert params["bandwidth_effective"] > 0
    assert debug.get("bandwidth_ms", {}).get("method") == "MS"


def test_ms_bandwidth_drives_bimodal_detection():
    rng = np.random.default_rng(123)
    data = np.concatenate([
        rng.normal(-2.5, 0.2, size=400),
        rng.normal(2.5, 0.25, size=400),
    ])

    peaks, valleys, xs, ys = kde_peaks_valleys(
        data,
        n_peaks=2,
        prominence=0.05,
        bw="MS",
        grid_size=6000,
        min_x_sep=0.5,
    )

    assert len(xs) > 0 and len(ys) > 0
    assert len(peaks) == 2
    assert len(valleys) == 1


def test_ms_bandwidth_works_with_workers():
    rng = np.random.default_rng(7)
    data = np.concatenate([
        rng.normal(-1.5, 0.3, size=300),
        rng.normal(1.75, 0.35, size=300),
    ])

    sample = batch_mod.SampleInput(
        stem="sample",
        counts=data,
        metadata={},
        arcsinh_signature=(False, 1.0, 0.2, 0.0),
    )
    options = BatchOptions(apply_arcsinh=False, bandwidth="MS", workers=2)

    results = batch_mod.run_batch([sample], options)
    assert results.samples

    debug = results.samples[0].params["debug"]["bandwidth_ms"]
    assert debug["method"] == "MS"
    assert debug["ms_workers"] >= 2
    assert results.samples[0].peaks
