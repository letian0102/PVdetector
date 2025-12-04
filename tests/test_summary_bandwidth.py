import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from peak_valley.batch import (  # noqa: E402
    BatchOptions,
    BatchResults,
    SampleInput,
    export_summary,
    process_sample,
)


def test_summary_reports_numeric_bandwidth_for_presets():
    counts = np.concatenate([
        np.random.default_rng(0).normal(-1.0, 0.1, 200),
        np.random.default_rng(1).normal(1.0, 0.1, 200),
    ])
    options = BatchOptions()
    sample = SampleInput(
        stem="demo",
        counts=counts,
        metadata={},
        arcsinh_signature=options.arcsinh_signature(),
    )

    result = process_sample(sample, options, overrides={}, gpt_client=None)
    batch = BatchResults(samples=[result])

    summary_df = export_summary(batch)
    bw_value = summary_df.loc[0, "bandwidth"]

    assert isinstance(bw_value, float)
    assert np.isfinite(bw_value)
    assert bw_value == result.params["bw"]


def test_fixed_bandwidth_is_preserved():
    counts = np.concatenate([
        np.random.default_rng(2).normal(-0.5, 0.05, 200),
        np.random.default_rng(3).normal(1.5, 0.05, 200),
    ])
    options = BatchOptions(bandwidth=0.2, apply_arcsinh=False)
    sample = SampleInput(
        stem="demo_bw",
        counts=counts,
        metadata={},
        arcsinh_signature=options.arcsinh_signature(),
    )

    result = process_sample(sample, options, overrides={}, gpt_client=None)

    assert np.isclose(result.params["bw"], 0.2)
    assert result.params.get("bw_label") == 0.2
