"""Command-line interface for the Peak & Valley detector."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import warnings
from pathlib import Path
from typing import Any

from peak_valley.batch import (
    BatchOptions,
    collect_counts_files,
    collect_dataset_samples,
    run_batch,
    save_outputs,
)

try:  # optional dependency – only needed when GPT features requested
    from openai import OpenAI
except Exception:  # pragma: no cover - OpenAI may not be installed
    OpenAI = None  # type: ignore


class ConsoleProgress:
    """Lightweight console progress bar for batch runs."""

    def __init__(self, stream: Any | None = None) -> None:
        self._stream = stream or sys.stderr
        self._lock = threading.Lock()
        self._total = 0
        self._completed = 0
        self._last_label: str | None = None
        self._active = False
        self._last_len = 0
        self._interactive = bool(getattr(self._stream, "isatty", lambda: False)())
        self._width = 28
        self._last_plain: str | None = None

    def start(self, total: int) -> None:
        with self._lock:
            self._total = max(int(total), 0)
            self._completed = 0
            self._last_label = None
            self._active = True
            self._last_plain = None
            if self._interactive and self._total > 0:
                message = self._format(None, 0, self._total, final=False, interrupted=False)
                self._emit(message, final=False)

    def advance(self, stem: str, completed: int, total: int) -> None:
        with self._lock:
            if not self._active:
                return
            self._total = max(self._total, int(total))
            self._completed = max(0, int(min(completed, max(total, completed))))
            self._last_label = stem
            message = self._format(stem, self._completed, self._total, final=False, interrupted=False)
            self._emit(message, final=False)

    def finish(self, completed: int, total: int, interrupted: bool) -> None:
        with self._lock:
            if not self._active:
                return
            self._total = max(self._total, int(total))
            self._completed = max(0, int(min(completed, max(total, completed))))
            message = self._format(self._last_label, self._completed, self._total, final=True, interrupted=interrupted)
            self._emit(message, final=True)
            self._active = False
            self._last_len = 0
            self._last_plain = None

    def _format(
        self,
        label: str | None,
        completed: int,
        total: int,
        *,
        final: bool,
        interrupted: bool,
    ) -> str:
        if total > 0:
            ratio = min(1.0, max(0.0, completed / max(total, 1)))
            filled = int(round(ratio * self._width))
            filled = min(self._width, max(0, filled))
            bar = "█" * filled + "░" * (self._width - filled)
            message = f"[{bar}] {completed}/{total}"
        else:
            message = f"Processed {completed} sample(s)"
        if label:
            message += f" ({label})"
        if final and interrupted:
            message += " — interrupted"
        return message

    def _emit(self, message: str, *, final: bool) -> None:
        if self._interactive:
            if len(message) < self._last_len:
                message = message + " " * (self._last_len - len(message))
            end = "\n" if final else "\r"
            print(message, end=end, file=self._stream, flush=True)
            self._last_len = 0 if final else len(message)
        else:
            if final and message == self._last_plain:
                return
            print(message, file=self._stream, flush=True)
            self._last_plain = message
def _parse_multi(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    items: list[str] = []
    for value in values:
        for part in value.split(","):
            clean = part.strip()
            if clean:
                items.append(clean)
    return items if items else None


def _normalize_all(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    lowered = {v.strip().lower() for v in values}
    if any(v in {"all", "*"} for v in lowered):
        return None
    return values


def _parse_batches(values: list[str] | None) -> list[str | None] | None:
    items = _parse_multi(values)
    if not items:
        return None
    converted: list[str | None] = []
    for value in items:
        lowered = value.lower()
        if lowered in {"none", "na", "null", "nan"}:
            converted.append(None)
        else:
            converted.append(value)
    return converted


def _load_overrides(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("Override file must contain a JSON object")
    return data


def _parse_target(values: str | None) -> list[float] | None:
    if not values:
        return None
    parts = [p.strip() for p in values.split(",") if p.strip()]
    if not parts:
        return None
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("Alignment target must be numeric values") from exc


def _configure_environment() -> None:
    """Silence noisy backend warnings and prefer a headless matplotlib backend."""

    # Avoid Qt initialisation when generating plots during batch runs.
    os.environ.setdefault("MPLBACKEND", "Agg")

    # Suppress the verbose CuPy experimental warning that surfaces in some
    # environments when cupyx.jit.rawkernel is imported indirectly.
    warnings.filterwarnings(
        "ignore",
        message="cupyx\\.jit\\.rawkernel is experimental",
        category=FutureWarning,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch processor for the Peak & Valley detector",
    )

    parser.add_argument(
        "--counts",
        action="append",
        metavar="PATH",
        help="Path to a *_raw_counts.csv file. May be provided multiple times.",
    )
    parser.add_argument(
        "--expression-file",
        help="expression_matrix_combined.csv for dataset mode.",
    )
    parser.add_argument(
        "--metadata-file",
        help="cell_metadata_combined.csv for dataset mode.",
    )
    parser.add_argument(
        "--marker",
        action="append",
        dest="markers",
        help="Marker(s) to analyse (comma separated or repeated).",
    )
    parser.add_argument(
        "--sample",
        action="append",
        dest="samples",
        help="Sample(s) to analyse (comma separated or repeated).",
    )
    parser.add_argument(
        "--batch",
        action="append",
        dest="batches",
        help="Batch values to include (comma separated). Use 'none' for empty batch.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where outputs will be written.",
    )
    parser.add_argument("--header-row", type=int, default=-1, help="Header row index for raw counts (−1 = none).")
    parser.add_argument("--skip-rows", type=int, default=0, help="Number of initial rows to skip in raw counts CSVs.")

    parser.add_argument("--arcsinh-a", type=float, default=1.0)
    parser.add_argument("--arcsinh-b", type=float, default=1 / 5)
    parser.add_argument("--arcsinh-c", type=float, default=0.0)
    parser.add_argument(
        "--apply-arcsinh",
        dest="apply_arcsinh",
        action="store_true",
        help="Apply arcsinh transform (default).",
    )
    parser.add_argument(
        "--skip-arcsinh",
        dest="apply_arcsinh",
        action="store_false",
        help="Skip arcsinh transform.",
    )
    parser.set_defaults(apply_arcsinh=None)

    parser.add_argument("--n-peaks", default=None, help="Fixed number of peaks or 'auto' for GPT/heuristic.")
    parser.add_argument("--max-peaks", type=int, default=3)
    parser.add_argument("--bandwidth", default="scott", help="Bandwidth rule/value or 'auto'.")
    parser.add_argument("--prominence", default="0.05", help="Prominence value or 'auto'.")
    parser.add_argument("--min-width", type=int, default=0)
    parser.add_argument("--curvature", type=float, default=0.0001)
    parser.add_argument("--turning-points", action="store_true", help="Count concave-down turning points as peaks.")
    parser.add_argument("--min-separation", type=float, default=0.7)
    parser.add_argument("--grid-size", type=int, default=20_000)
    parser.add_argument("--valley-drop", type=float, default=10.0, help="Valley drop threshold in percent.")
    parser.add_argument(
        "--first-valley",
        choices=["slope", "drop"],
        default="slope",
        help="Method for the first valley after the leading peak.",
    )

    parser.add_argument("--consistency-tol", type=float, default=0.5, help="Tolerance for marker consistency enforcement.")
    parser.add_argument("--consistency", dest="consistency", action="store_true", help="Enforce marker consistency.")
    parser.add_argument("--no-consistency", dest="consistency", action="store_false", help="Disable marker consistency.")
    parser.set_defaults(consistency=None)

    parser.add_argument("--align", action="store_true", help="Align and normalise distributions after detection.")
    parser.add_argument(
        "--alignment-mode",
        choices=["negPeak", "negPeak_valley", "negPeak_valley_posPeak", "valley"],
        default="negPeak_valley_posPeak",
    )
    parser.add_argument(
        "--alignment-target",
        help="Comma-separated landmark targets (otherwise cohort median is used).",
    )

    parser.add_argument("--workers", type=int, default=1, help="Number of parallel worker threads.")
    parser.add_argument("--override-file", help="JSON file with per-sample or per-marker overrides.")
    parser.add_argument("--gpt-model", help="Custom OpenAI model name for GPT-assisted suggestions.")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="OpenAI API key for GPT-assisted suggestions (overrides OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--export-plots",
        action="store_true",
        help="Also write per-sample density plots into an output 'plots' folder.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_environment()

    parser = build_parser()
    args = parser.parse_args(argv)

    counts_paths = args.counts or []
    markers = _normalize_all(_parse_multi(args.markers))
    samples = _normalize_all(_parse_multi(args.samples))
    batches = _parse_batches(args.batches)
    target_landmarks = _parse_target(args.alignment_target)
    overrides = _load_overrides(args.override_file)

    options = BatchOptions()
    options.arcsinh_a = args.arcsinh_a
    options.arcsinh_b = args.arcsinh_b
    options.arcsinh_c = args.arcsinh_c
    options.apply_arcsinh = True if args.apply_arcsinh is None else bool(args.apply_arcsinh)
    options.export_plots = bool(args.export_plots)

    options.max_peaks = max(1, args.max_peaks)
    options.min_width = max(0, args.min_width)
    options.curvature = args.curvature
    options.turning_points = bool(args.turning_points)
    options.min_separation = args.min_separation
    options.grid_size = max(4000, args.grid_size)
    options.valley_drop = args.valley_drop
    options.first_valley = args.first_valley
    options.consistency_tol = args.consistency_tol
    if args.consistency is not None:
        options.apply_consistency = bool(args.consistency)

    options.align = bool(args.align)
    options.align_mode = args.alignment_mode
    options.target_landmarks = target_landmarks
    options.workers = max(1, args.workers)

    if args.gpt_model:
        options.gpt_model = args.gpt_model

    # handle n-peaks / GPT settings
    if args.n_peaks is None:
        options.n_peaks = None
        options.n_peaks_auto = True
    else:
        text = str(args.n_peaks).strip()
        if text.lower() in {"auto", "gpt"}:
            options.n_peaks = None
            options.n_peaks_auto = True
        else:
            try:
                options.n_peaks = int(float(text))
                options.n_peaks_auto = False
            except ValueError:
                parser.error("--n-peaks must be an integer or 'auto'")

    bw_text = str(args.bandwidth).strip()
    if bw_text.lower() in {"auto", "gpt"}:
        options.bandwidth_auto = True
    else:
        try:
            options.bandwidth = float(bw_text) if bw_text else "scott"
        except ValueError:
            options.bandwidth = bw_text or "scott"
    prom_text = str(args.prominence).strip()
    if prom_text.lower() in {"auto", "gpt"}:
        options.prominence_auto = True
    else:
        try:
            options.prominence = float(prom_text)
        except ValueError:
            parser.error("--prominence must be numeric or 'auto'")

    samples_inputs = []
    run_metadata: dict[str, Any] = {}

    if counts_paths:
        counts_inputs = collect_counts_files(
            [Path(p) for p in counts_paths],
            options,
            header_row=args.header_row,
            skip_rows=args.skip_rows,
        )
        samples_inputs.extend(counts_inputs)

    if args.expression_file and args.metadata_file:
        dataset_samples, dataset_meta = collect_dataset_samples(
            args.expression_file,
            args.metadata_file,
            options,
            markers=markers or None,
            samples_filter=samples or None,
            batches=batches,
        )
        run_metadata.update(dataset_meta)
        samples_inputs.extend(dataset_samples)
    elif args.expression_file or args.metadata_file:
        parser.error("Both --expression-file and --metadata-file are required for dataset mode.")

    if not samples_inputs:
        parser.error("Provide at least one --counts file or a dataset.")

    for idx, sample in enumerate(samples_inputs):
        sample.order = idx

    need_gpt = options.bandwidth_auto or options.prominence_auto or options.n_peaks_auto
    gpt_client = None
    if need_gpt:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "[warning] GPT-enabled options requested but no API key was provided via --api-key or OPENAI_API_KEY. "
                "Falling back to defaults.",
                file=sys.stderr,
            )
        elif OpenAI is None:
            print("[warning] openai package is not available. GPT features will be skipped.", file=sys.stderr)
        else:
            try:
                gpt_client = OpenAI(api_key=api_key)
            except Exception as exc:  # pragma: no cover - depends on environment
                print(f"[warning] Failed to initialise OpenAI client: {exc}", file=sys.stderr)

    total_requested = len(samples_inputs)
    progress = ConsoleProgress()

    batch = run_batch(
        samples_inputs,
        options,
        overrides,
        gpt_client,
        progress=progress,
    )
    save_outputs(
        batch,
        args.output_dir,
        run_metadata=run_metadata,
        export_plots=options.export_plots,
    )

    processed = len(batch.samples)
    if batch.interrupted:
        print(
            f"[warning] Processing interrupted after {processed} of {total_requested} sample(s). "
            f"Partial results saved to {args.output_dir}.",
            file=sys.stderr,
        )
        return 130

    print(f"Processed {processed} sample(s). Results saved to {args.output_dir}.")
    return 0
if __name__ == "__main__":
    sys.exit(main())
