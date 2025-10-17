"""Utility package for the Peak-&-Valley Streamlit app."""
# re-export the high-level helpers so main_app can import them flat
from .data_io       import arcsinh_transform, read_counts, load_combined_csv
from .kde_detector   import kde_peaks_valleys, quick_peak_estimate
from .plotting       import fig_to_png, thumb64
from .gpt_adapter    import ask_gpt_peak_count           # noqa: F401
from .consistency    import enforce_marker_consistency    # noqa: F401
from .backend        import backend_description, get_array_backend  # noqa: F401
