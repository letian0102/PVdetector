import time
from collections import defaultdict

import numpy as np

from peak_valley import batch


def _make_sample(stem: str, order: int = 0) -> batch.SampleInput:
    counts = np.array([1.0, 2.0, 3.0])
    metadata: dict[str, object] = {}
    return batch.SampleInput(
        stem=stem,
        counts=counts,
        metadata=metadata,
        arcsinh_signature=(True, 1.0, 0.2, 0.0),
        order=order,
    )


def _make_result(stem: str) -> batch.SampleResult:
    counts = np.array([1.0, 2.0, 3.0])
    xs = np.array([0.0, 1.0, 2.0])
    ys = np.array([0.1, 0.2, 0.3])
    return batch.SampleResult(
        stem=stem,
        peaks=[1.0],
        valleys=[0.5],
        xs=xs,
        ys=ys,
        counts=counts,
        params={},
        quality=1.0,
        metadata={},
    )


def test_run_batch_reschedules_timed_out_workers(monkeypatch):
    attempts: defaultdict[str, int] = defaultdict(int)

    def fake_process_sample(sample: batch.SampleInput, *_args, **_kwargs):
        attempts[sample.stem] += 1
        if sample.stem == "slow" and attempts[sample.stem] < 3:
            time.sleep(0.15)
        return _make_result(sample.stem)

    monkeypatch.setattr(batch, "process_sample", fake_process_sample)

    samples = [_make_sample("slow"), _make_sample("fast1", order=1), _make_sample("fast2", order=2)]
    options = batch.BatchOptions(workers=2, worker_timeout=0.1)

    results = batch.run_batch(samples, options)

    assert len(results.samples) == 3
    assert attempts["slow"] == 3
