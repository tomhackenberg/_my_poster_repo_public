# Measuring What Matters: A Forward-Engineered Approach to Causal Inference with Endogenous Staggered Adoption

**Supplementary Materials**

This repository contains the complete analysis code and pre-computed results for the research presented at [CODE@MIT] on November 15, 2025.

> **Note on Data:** The raw data for this project is confidential and not included in this repository. However, all scripts, code, and final outputs are provided for full transparency and verification of the analytical workflow.

---

## 📁 Repository Structure

```
├── src/                    # Source code (Python .py and Quarto .qmd files)
├── results/                # Pre-computed tables (.csv) by sensitivity threshold
│   ├── 70pct/             # Results for 70% threshold
│   ├── 80pct/             # Primary results (80% threshold)
│   └── 90pct/             # Results for 90% threshold
├── figures/                # Pre-computed plots (.png) by sensitivity threshold
│   ├── 70pct/
│   ├── 80pct/
│   └── 90pct/
├── docs/                   # Extended abstract, references.bib, and supporting docs
├── data/                   # (Empty) Placeholder for private raw data
├── requirements.txt        # Python dependencies
└── renv.lock              # R package dependencies
```

---

## 🔍 Navigating the Results

While the full pipeline cannot be re-run without access to the private data, you can inspect all code and pre-computed outputs.

### Key Files

**Primary Analysis (80% threshold):**
- Summary table: `./results/80pct/all_results_summary.csv`
- Rendered report: `./results/80pct/02_analysis_revised.html`
- Figures: `./figures/80pct/`

**Sensitivity Analyses:**
- 70% threshold: `./results/70pct/` and `./figures/70pct/`
- 90% threshold: `./results/90pct/` and `./figures/90pct/`

---

## 🔄 Analysis Pipeline

The workflow consists of three sequential steps located in `/src/`:

### Step 1: Data Preparation
**`01_revised_data_preparation.py`**
- Processes raw data (private)
- Implements effect onset detection algorithm
- Generates intermediate `.parquet` analysis datasets

### Step 2: Causal Analysis
**`02_analysis_revised.qmd`**
- Loads `.parquet` files from Step 1
- Runs causal models (DiD, HonestDiD)
- Saves model objects, plots, and result tables

### Step 3: Meta-Analysis
**`03_analysis_revised.qmd`**
- Loads results from Step 2
- Generates meta-analysis plots and summary tables

---

## 🚀 Replication Instructions

### Prerequisites

Install dependencies:

```bash
# Python packages
pip install -r requirements.txt

# R packages
Rscript -e "renv::restore()"
```

### Execution

To generate outputs for the primary analysis (80% threshold), run the following commands in order:

#### 1. Run Data Preparation Scripts (9 commands)

**Main Analysis:**
```bash
python src/01_revised_data_preparation.py --method effect_onset --onset_threshold 0.8 --placebo_type none
```

**Pre-Period Placebos (5 verticals):**
```bash
python src/01_revised_data_preparation.py --method effect_onset --vertical tech --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical economie --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical achterklap --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical Media_en_Cultuur --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical goed_nieuws --onset_threshold 0.8 --placebo_type pre_period
```

**Additional Analyses:**
```bash
python src/01_revised_data_preparation.py --method artificial_stagger --vertical tech
python src/01_revised_data_preparation.py --method naive_launch
python src/01_revised_data_preparation.py --method thematic_tiers
```

#### 2. Render Analysis Reports (2 commands)

```bash
quarto render src/02_analysis_revised.qmd --execute-param onset_pct:80pct
quarto render src/03_analysis_revised.qmd --execute-param onset_pct:80pct
```

### Sensitivity Analyses

To generate results for alternative thresholds (70% or 90%), repeat the entire process with adjusted parameters:

- For 70%: use `--onset_threshold 0.7` and `onset_pct:70pct`
- For 90%: use `--onset_threshold 0.9` and `onset_pct:90pct`

---

## 📊 Output Files

All pre-computed results are organized by threshold:

- **Tables:** `./results/{threshold}/`
- **Figures:** `./figures/{threshold}/`
- **HTML Reports:** `./results/{threshold}/*.html`

---

## 📝 Citation

If you use this code or methodology, please cite:

```bibtex
[Add your citation here]
```

---

## 📧 Contact

For questions or issues, please [open an issue](../../issues) or contact [your contact information].

---

## 📄 License

[Specify your license here]