# PVdetector

PVdetector is a Streamlit application for detecting density peaks and valleys in single-cell protein count data, aligning markers across samples, and exporting normalized distributions.

## Features
- Upload raw counts or full datasets (expression matrix + metadata).
- Automatic or manual control over KDE bandwidth, peak count, and prominence. GPT-based suggestions are available when an OpenAI API key is provided.
- Interactive per-sample visualization with manual editing of peaks and valleys.
- Optional enforcement of marker consistency across samples.
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

## Usage
### A. Upload counts CSV files
- Upload one or more `*_raw_counts.csv` files, each containing a single column of numeric counts.
- The files are cached and selectable for processing.

### B. Upload whole dataset
- Upload `expression_matrix_combined.csv` and `cell_metadata_combined.csv`.
- Choose markers, samples, and optional batches; the app generates per-sample counts files for analysis.

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

