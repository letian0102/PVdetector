# PVdetector

PVdetector is a Streamlit application for detecting density peaks and valleys in single-cell protein count data, aligning markers across samples, and exporting normalized distributions.

## Features
- Upload raw counts or full datasets (expression matrix + metadata).
- Optional arcsinh transformation with adjustable parameters \(a, b, c\).
- Automatic or manual control over KDE bandwidth, peak count, and prominence. GPT-based suggestions (bandwidth scans multiple candidates for optimal peak separation) are available when an OpenAI API key is provided.
- Interactive per-sample visualization with manual editing of peaks and valleys.
- Optional enforcement of marker consistency across samples (disabled by default for batch runs).
- Landmark alignment and piece-wise linear normalization across samples.
- Downloadable outputs: per-sample curves, aligned data, stain-quality scores, and summary tables.

## Installation
1. **Clone the repository**
    ```bash
    git clone <REPO_URL>
    cd PVdetector
    ```
2. **Create a virtual environment and install dependencies**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
3. **OpenAI API key (optional)**
   - Required for GPT-assisted parameter suggestions. Set the `OPENAI_API_KEY` environment variable or enter the key in the app.

## Running the app
1. Start the Streamlit server from the project root:
    ```bash
    streamlit run app.py
    ```
2. A browser window will open. Avoid refreshing or closing the tab—uploaded data lives only in memory.

## Batch processing from the command line
The repository now includes a non-interactive workflow that mirrors the Streamlit
features, making it easy to run analyses on a server or inside scheduled jobs.

```bash
python batch_run.py \
    --expression-file expression_matrix_combined.csv \
    --metadata-file cell_metadata_combined.csv \
    --marker CD3 \
    --sample SampleA,SampleB \
    --align --output-dir results_batch
```

Key flags:

- `--counts`: process individual `*_raw_counts.csv` files (repeatable).
- `--expression-file` / `--metadata-file`: analyse a whole dataset; combine with `--marker`, `--sample`, and `--batch` to filter selections. Pass `--marker all` or `--sample all` to process every marker/sample without enumerating them.
- `--override-file`: JSON mapping of global/marker/sample/stem overrides, e.g.
  ```json
  {
    "markers": {"CD3": {"n_peaks": 2}},
    "stems": {"SampleA_CD3_raw_counts": {"bandwidth": 0.8}}
  }
  ```

### CLI parameter reference

| Parameter | Example usage | Accepted values / notes |
|-----------|---------------|-------------------------|
| `--counts` | `--counts SampleA_CD3_raw_counts.csv --counts SampleB_CD4_raw_counts.csv` | Path(s) to individual counts CSV files; repeat per file. |
| `--expression-file` | `--expression-file expression_matrix_combined.csv` | Combined expression matrix for dataset mode; use with `--metadata-file`. |
| `--metadata-file` | `--metadata-file cell_metadata_combined.csv` | Metadata CSV that pairs with `--expression-file`. |
| `--marker` | `--marker CD3` | Comma-separated or repeatable marker list; use `--marker all`/`*` for every marker. |
| `--sample` | `--sample SampleA,SampleB` | Comma-separated or repeatable sample list; use `--sample all`/`*` for every sample. |
| `--batch` | `--batch B1 --batch none` | Comma-separated or repeatable batch filters; `none`/`null` selects missing batches. |
| `--output-dir` | `--output-dir results_batch` | Destination folder; created if absent. |
| `--header-row` | `--header-row -1` | Header row index for raw counts (`-1` = no header). |
| `--skip-rows` | `--skip-rows 2` | Number of initial lines to ignore in counts CSVs. |
| `--apply-arcsinh` / `--skip-arcsinh` | `--skip-arcsinh` | Toggle arcsinh preprocessing; default is to apply it. |
| `--arcsinh-a` | `--arcsinh-a 1.5` | Positive scaling parameter `a` for arcsinh transform. |
| `--arcsinh-b` | `--arcsinh-b 0.2` | Scaling parameter `b`; provide a decimal float. |
| `--arcsinh-c` | `--arcsinh-c -1.0` | Offset parameter `c`; may be negative. |
| `--n-peaks` | `--n-peaks 2` | Fixed peak count or `auto`/`gpt` for automatic selection. |
| `--max-peaks` | `--max-peaks 4` | Upper bound on peaks considered during automatic runs. |
| `--bandwidth` | `--bandwidth 0.6` | KDE bandwidth value, preset (`scott`, `silverman`), or `auto`/`gpt`. |
| `--prominence` | `--prominence 0.08` | Minimum prominence (`auto`/`gpt` allowed). |
| `--min-width` | `--min-width 30` | Minimum sample count per detected peak. |
| `--curvature` | `--curvature 0.0005` | Curvature threshold for peak detection. |
| `--turning-points` | `--turning-points` | Flag to treat concave-down turning points as peaks. |
| `--min-separation` | `--min-separation 0.5` | Minimum distance between peaks in marker units. |
| `--grid-size` | `--grid-size 40000` | Number of KDE evaluation points (minimum 4000). |
| `--valley-drop` | `--valley-drop 12.5` | Percent drop required between peaks. |
| `--first-valley` | `--first-valley drop` | Strategy for the first valley: `slope` or `drop`. |
| `--apply-consistency-match` / `--skip-consistency-match` | `--apply-consistency-match` | Toggle cross-sample marker consistency; disabled by default. |
| `--consistency-tol` | `--consistency-tol 0.3` | Allowed variation when consistency matching is enabled. |
| `--align` | `--align` | Enable landmark alignment and normalisation. |
| `--alignment-mode` | `--alignment-mode negPeak_valley` | Alignment template: `negPeak`, `negPeak_valley`, `negPeak_valley_posPeak`, or `valley`. |
| `--alignment-target` | `--alignment-target -1.5,0.0,1.5` | Comma-separated landmark targets; defaults to cohort medians. |
| `--workers` | `--workers 4` | Number of parallel worker threads. |
| `--override-file` | `--override-file overrides.json` | JSON overrides for global/marker/sample/stem settings. |
| `--gpt-model` | `--gpt-model gpt-4o-mini` | Override GPT model name when using automatic suggestions. |
| `--api-key` | `--api-key sk-...` | OpenAI API key; overrides the `OPENAI_API_KEY` environment variable. |
| `--export-plots` | `--export-plots` | Emit per-sample ridge PNGs inside a `plots/` folder. |

By default, outputs in `--output-dir` comprise `summary.csv`, `results.json`,
and the `before_after_alignment.zip` bundle that mirrors the Streamlit
"before/after" download (combined metadata/expression CSVs plus stacked ridge
plots, without per-sample CSVs). Pass `--export-plots` to add a `plots/`
directory containing the individual density visuals.

The CLI prints a progress bar while samples are processed and automatically
finalises partial results if an analysis is interrupted.

## Usage
### A. Upload counts CSV files
- Upload one or more `*_raw_counts.csv` files, each containing a single column of numeric counts.
- The files are cached and selectable for processing.
- Use the **Preprocessing** section to apply or skip an arcsinh transform and tweak parameters.

### B. Upload whole dataset
- Upload `expression_matrix_combined.csv` and `cell_metadata_combined.csv`.
- Choose markers, samples, and optional batches; the app generates per-sample counts files for analysis.
- The **Preprocessing** section controls whether these counts are arcsinh-transformed and allows customization of \(a, b, c\).

### Detection settings
- **Number of peaks** – enter a fixed value or select “GPT Automatic” with a user-defined maximum.
- **Bandwidth** and **Prominence** – choose manual presets/scales or allow GPT to suggest values.
- Additional controls include minimum peak width, curvature threshold, concave turning points, minimum separation, KDE grid size, valley drop, first-valley method selection, and marker-consistency enforcement.

### Running the detector
- Click **Run detector** to process selected files. A progress bar tracks the queue, and a Pause/Resume button provides mid-run control.
- Use **Clear results** to reset all session data.

### Viewing and editing results
Each processed sample appears under **Processed datasets** with three tabs:
- **Plot** – density curve with detected peaks/valleys and stain-quality score.
- **Parameters** – adjust bandwidth, prominence, and peak count per sample. Modified samples can be reprocessed.
- **Manual** – interactively add, move, or delete peaks and valleys.

### Alignment & normalization
- Choose a landmark set and specify target positions (automatic or manual).
- After detection, click **Align landmarks & normalize counts** to apply piece-wise linear warping across samples.
- A comparison tab shows raw versus aligned ridge plots when alignment results exist.

### Downloads
Under **Summary ∣ downloads**, obtain:
- `SampleCurves.zip` – per-sample `xs`/`ys` curves.
- `alignedData.zip` – available after alignment.
- `StainQuality.csv` – per-sample stain-quality scores.
- `PeakValleyResults.zip` – summary CSV plus all plots.

## Development
Run unit tests with:
```bash
pytest
```

## Contributing
Issues and pull requests are welcome. Please include tests for new features or bug fixes.

