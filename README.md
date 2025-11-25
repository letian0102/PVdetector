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
- When selecting all markers or all samples (in the app or CLI), you can exclude specific markers/samples from processing.
- Configurable per-sample timeout in batch mode that skips long-running samples and records them in the output folder.

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
- `--expression-file` / `--metadata-file`: analyse a whole dataset; combine with `--marker`, `--sample`, and `--batch` to filter selections. Pass `--marker all` or `--sample all` to process every marker/sample without enumerating them; combine with the exclude flags below when needed.
- `--override-file`: JSON mapping of global/marker/sample/stem overrides, e.g.
  ```json
  {
    "markers": {"CD3": {"n_peaks": 2}},
    "stems": {"SampleA_CD3_raw_counts": {"bandwidth": 0.8}}
  }
  ```

### CLI parameter reference

| Parameter | Required? | Default | Example usage | Accepted values / notes |
|-----------|-----------|---------|---------------|-------------------------|
| `--counts` | Optional¹ | _None_ | `--counts SampleA_CD3_raw_counts.csv --counts SampleB_CD4_raw_counts.csv` | Path(s) to individual counts CSV files; repeat per file. |
| `--expression-file` | Optional¹ | _None_ | `--expression-file expression_matrix_combined.csv` | Combined expression matrix for dataset mode; must be paired with `--metadata-file`. |
| `--metadata-file` | Optional¹ | _None_ | `--metadata-file cell_metadata_combined.csv` | Metadata CSV for dataset mode; required when `--expression-file` is supplied. |
| `--marker` | Optional | All markers | `--marker CD3` | Comma-separated or repeatable marker list; use `--marker all`/`*` for every marker. |
| `--sample` | Optional | All samples | `--sample SampleA,SampleB` | Comma-separated or repeatable sample list; use `--sample all`/`*` for every sample. |
| `--exclude-marker` | Optional | _None_ | `--marker all --exclude-marker CD3,CD4` | Marker(s) to skip when processing all markers; comma-separated or repeatable. |
| `--exclude-sample` | Optional | _None_ | `--sample all --exclude-sample SampleX` | Sample(s) to skip when processing all samples; comma-separated or repeatable. |
| `--batch` | Optional | No batch filter | `--batch B1 --batch none` | Comma-separated or repeatable batch filters; `none`/`null` selects missing batches. |
| `--output-dir` | **Required** | — | `--output-dir results_batch` | Destination folder; created if absent. |
| `--header-row` | Optional | `-1` | `--header-row -1` | Header row index for raw counts (`-1` = no header). |
| `--skip-rows` | Optional | `0` | `--skip-rows 2` | Number of initial lines to ignore in counts CSVs. |
| `--apply-arcsinh` / `--skip-arcsinh` | Optional | Apply arcsinh | `--skip-arcsinh` | Toggle arcsinh preprocessing; default is to apply it. |
| `--arcsinh-a` | Optional | `1.0` | `--arcsinh-a 1.5` | Positive scaling parameter `a` for arcsinh transform. |
| `--arcsinh-b` | Optional | `0.2` | `--arcsinh-b 0.2` | Scaling parameter `b`; provide a decimal float. |
| `--arcsinh-c` | Optional | `0.0` | `--arcsinh-c -1.0` | Offset parameter `c`; may be negative. |
| `--n-peaks` | Optional | Auto | `--n-peaks 2` | Fixed peak count or `auto`/`gpt` for automatic selection. |
| `--max-peaks` | Optional | `2` | `--max-peaks 4` | Upper bound on peaks considered during automatic runs. |
| `--bandwidth` | Optional | `scott` | `--bandwidth 0.6` | KDE bandwidth value, preset (`scott`, `silverman`), or `auto`/`gpt`. |
| `--prominence` | Optional | `0.05` | `--prominence 0.08` | Minimum prominence (`auto`/`gpt` allowed). |
| `--min-width` | Optional | `0` | `--min-width 30` | Minimum sample count per detected peak. |
| `--curvature` | Optional | `0.0001` | `--curvature 0.0005` | Curvature threshold for peak detection. |
| `--turning-points` | Optional | Off | `--turning-points` | Flag to treat concave-down turning points as peaks. |
| `--min-separation` | Optional | `0.5` | `--min-separation 0.5` | Minimum distance between peaks in marker units. |
| `--grid-size` | Optional | `20000` | `--grid-size 40000` | Number of KDE evaluation points (minimum 4000). |
| `--valley-drop` | Optional | `10.0` | `--valley-drop 12.5` | Percent drop required between peaks. |
| `--first-valley` | Optional | `slope` | `--first-valley drop` | Strategy for the first valley: `slope` or `drop`. |
| `--peak-model` | Optional | _None_ | `--peak-model weights.joblib` | Path to a saved peak/non-peak classifier; when omitted, heuristics are used. |
| `--peak-model-threshold` | Optional | `0.6` | `--peak-model-threshold 0.7` | Minimum probability required for a model-proposed peak. |
| `--peak-model-confidence` | Optional | `0.55` | `--peak-model-confidence 0.6` | If the model's best score is below this value, the detector falls back to the heuristic path. |
| `--apply-consistency-match` / `--skip-consistency-match` | Optional | Skip consistency | `--apply-consistency-match` | Toggle cross-sample marker consistency; disabled by default. |
| `--consistency-tol` | Optional | `0.5` | `--consistency-tol 0.3` | Allowed variation when consistency matching is enabled. |
| `--align` | Optional | Off | `--align` | Enable landmark alignment and normalisation. |
| `--alignment-mode` | Optional | `negPeak_valley_posPeak` | `--alignment-mode negPeak_valley` | Alignment template: `negPeak`, `negPeak_valley`, `negPeak_valley_posPeak`, or `valley`. |
| `--alignment-target` | Optional | Cohort medians | `--alignment-target -1.5,0.0,1.5` | Comma-separated landmark targets; defaults to cohort medians. |
| `--sample-timeout` | Optional | `10.0` | `--sample-timeout 5` | Maximum seconds to process each sample before skipping it (`<=0` disables). Timed-out samples are listed in `error_samples.txt`. |
| `--workers` | Optional | `1` | `--workers 4` | Number of parallel worker threads. |
| `--override-file` | Optional | _None_ | `--override-file overrides.json` | JSON overrides for global/marker/sample/stem settings. |
| `--gpt-model` | Optional | Auto (`o4-mini`) | `--gpt-model o4-mini` | Override GPT model name when using automatic suggestions. |
| `--api-key` | Optional | Environment variable | `--api-key sk-...` | OpenAI API key; overrides the `OPENAI_API_KEY` environment variable. |
| `--export-plots` | Optional | Off | `--export-plots` | Emit per-sample ridge PNGs inside a `plots/` folder. |

¹ Provide at least one `--counts` file or the `--expression-file`/`--metadata-file` pair.

By default, outputs in `--output-dir` comprise `summary.csv`, `results.json`,
`error_samples.txt` (one stem per timed-out/failed sample), and the
`before_after_alignment.zip` bundle that mirrors the Streamlit "before/after"
download (combined metadata/expression CSVs plus stacked ridge plots, without
per-sample CSVs). Pass `--export-plots` to add a `plots/` directory containing
the individual density visuals.

The CLI prints a progress bar while samples are processed and automatically
finalises partial results if an analysis is interrupted.

## Usage
### A. Upload counts CSV files
- Upload one or more `*_raw_counts.csv` files, each containing a single column of numeric counts.
- The files are cached and selectable for processing.
- Use the **Preprocessing** section to apply or skip an arcsinh transform and tweak parameters.

### B. Upload whole dataset
- Upload `expression_matrix_combined.csv` and `cell_metadata_combined.csv`.
- Choose markers, samples, and optional batches; the app generates per-sample counts files for analysis. Selecting “all markers” or “all samples” reveals exclude pickers so you can omit specific markers/samples without enumerating the rest.
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

### Training the optional peak scorer
The detector can bootstrap peaks from a lightweight gradient-boosted classifier
that scores each KDE grid cell using local windows of the density curve.

1. Create labelled traces (binary arrays aligned to the KDE grid where `1` marks
   a peak location).
2. Fit and persist the model:
   ```python
   from peak_valley.peak_model import GradientBoostingPeakScorer

   scorer = GradientBoostingPeakScorer(window_radius=5)
   scorer.fit([trace1, trace2], [labels1, labels2])
   scorer.save("weights/peaks.joblib")
   ```
3. Run the CLI with `--peak-model weights/peaks.joblib` to enable the
   probability-driven candidate peaks. The detector will automatically fall
   back to the existing heuristics when the model's best score drops below the
   configured confidence threshold.

## Contributing
Issues and pull requests are welcome. Please include tests for new features or bug fixes.

