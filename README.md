# Measuring What Matters: A Forward-Engineered Approach to Causal Inference with Endogenous Staggered Adoption

This repository contains the full analysis code, figures, and pre-computed results for the research presented at **CODE@MIT on November 15, 2025**.

The **live, interactive HTML reports** (which show all code and outputs) are the best way to review this analysis.

> **Note on Data:** The raw data for this project is confidential and not included in this repository. However, all scripts and outputs are provided for full transparency of the analytical workflow.

---

## 🚀 View the Live Analysis Reports

This repository is hosted on GitHub Pages. You can view the final, rendered HTML reports directly in your browser.

* **[View 80% Threshold Analysis (Primary)](https://tomhackenberg.github.io/_my_poster_repo_public/src/results/80pct/02_analysis_revised.html)**
* **[View 70% Threshold Analysis (Sensitivity)](https://tomhackenberg.github.io/_my_poster_repo_public/src/results/70pct/02_analysis_revised.html)**
* **[View 90% Threshold Analysis (Sensitivity)](https://tomhackenberg.github.io/_my_poster_repo_public/src/results/90pct/02_analysis_revised.html)**

---

## 🔬 The 3-Step Analysis Pipeline

The full workflow is contained in the `/src/` folder:

1.  **`01_revised_data_preparation.py`**
    * This Python script implements our custom **effect onset detection algorithm**. It processes the (private) raw data to find the *true* treatment date for each article, generating the `.parquet` analysis datasets.

2.  **`02_analysis_revised.qmd`**
    * This Quarto report runs the core causal analysis. It loads the `.parquet` files and uses the `did` package in R to run the staggered Difference-in-Differences models, robustness checks, and HonestDiD.

3.  **`03_analysis_revised.qmd`**
    * This final Quarto report loads the saved model results from Step 2 to generate the meta-analysis plots (like the "Format vs. Topic" mechanism) and summary tables.

---

<details>
<summary><b>Click here for Full Replication Instructions & Repository Contents</b></summary>

### 🔄 Replication Instructions

While you cannot re-run the pipeline without the private data, here is the full set of commands used to generate all outputs for a single run (e.g., 80% threshold).

#### 1. Prerequisites

```bash
# Install Python packages
pip install -r requirements.txt

# Install R packages
Rscript -e "renv::restore()"
```

#### 2. Run Data Preparation (9 Python scripts)
These commands must be run from the repository's root directory.

```bash
# 1. Main Analysis
python src/01_revised_data_preparation.py --method effect_onset --onset_threshold 0.8 --placebo_type none

# 2. Pre-Period Placebos
python src/01_revised_data_preparation.py --method effect_onset --vertical tech --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical economie --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical achterklap --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical Media_en_Cultuur --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical goed_nieuws --onset_threshold 0.8 --placebo_type pre_period

# 3. Other Analyses
python src/01_revised_data_preparation.py --method artificial_stagger --vertical tech
python src/01_revised_data_preparation.py --method naive_launch
python src/01_revised_data_preparation.py --method thematic_tiers
```

#### 3. Render Analysis Reports (2 Quarto commands)
These commands (run from the root) generate all figures, CSVs, and the final HTML reports (which are created inside src/).

```bash
quarto render src/02_analysis_revised.qmd --execute-param onset_pct:80pct
quarto render src/03_analysis_revised.qmd --execute-param onset_pct:80pct
```


## 📁 Repository Structure

├── src/                    # Source code (Python .py and Quarto .qmd files)
├── results/                # Pre-computed tables (.csv) and HTML reports
│   ├── 70pct/
│   ├── 80pct/
│   └── 90pct/
├── figures/                # Pre-computed plots (.png)
│   ├── 70pct/
│   ├── 80pct/
│   └── 90pct/
├── docs/                   # Extended abstract, references.bib
├── data/                   # (Empty) Placeholder for private raw data
├── requirements.txt        # Python dependencies
└── renv.lock              # R package dependencies