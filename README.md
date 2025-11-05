Supplementary Materials for: Measuring What Matters: A Forward-Engineered Approach to Causal Inference with Endogenous Staggered Adoption

This repository contains the full analysis code and pre-computed results for the analysis presented at:
[CODE@MIT] on [11.15.2025].

The raw data for this project is private and is not included. However, this repository provides all scripts, code, and final outputs for full transparency and verification of the workflow.

1. Repository Structure

/src/: All source code, including Python (.py) and Quarto (.qmd) analysis scripts.

/results/: All pre-computed tabular outputs (e.g., .csv files) for the three different sensitivity thresholds (70pct, 80pct, 90pct).

/figures/: All pre-computed plots (e.g., .png files) for the three sensitivity thresholds.

/docs/: The extended abstract, references.bib, and other supporting documents.

/data/: This folder is intentionally empty in the repository. The original raw data is private and was stored here locally to run the analysis.

requirements.txt: A list of required Python packages (pip install -r requirements.txt).

renv.lock: A list of required R packages (renv::restore()).

2. How to Navigate These Results

While you cannot re-run the full pipeline without the private data, you can inspect the complete code and all of its final, pre-computed outputs.

The analysis code that generates all results is in /src/.

The final outputs (tables and figures) are in /results/ and /figures/, organized by the onset detection threshold:

./results/80pct/: The primary results (80% onset threshold).

./figures/80pct/: The primary plots (80% onset threshold).

./results/70pct/ & ./figures/70pct/: Sensitivity analysis (70% threshold).

./results/90pct/ & ./figures/90pct/: Sensitivity analysis (90% threshold).

For example, the main summary table for the primary analysis is located at:
./results/80pct/all_results_summary.csv

The rendered HTML reports, which show the code and output side-by-side, are also in their respective folders (e.g., ./results/80pct/02_analysis_revised.html).

3. Workflow & Replication

The analysis is a 3-step pipeline found in /src/:

01_revised_data_preparation.py: This script takes the (private) raw data, runs the core effect onset detection algorithm, and generates all intermediate .parquet analysis datasets.

02_analysis_revised.qmd: This Quarto document loads the .parquet files, runs all causal models (DiD, HonestDiD), and saves all model objects, plots, and result tables.

03_analysis_revised.qmd: This final Quarto document loads the results from step 2 to generate the meta-analysis plots and summary tables.

Execution Commands

To generate all outputs for a single sensitivity run (e.g., the primary 0.8 threshold), you must execute the following 11 commands from your terminal in this order.

Step 1: Run the 9 Python Data Prep Scripts

# === 1. Main Analysis (1 command for all 5 verticals) ===
python src/01_revised_data_preparation.py --method effect_onset --onset_threshold 0.8 --placebo_type none

# === 2. Pre-Period Placebos (5 commands) ===
python src/01_revised_data_preparation.py --method effect_onset --vertical tech --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical economie --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical achterklap --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical Media_en_Cultuur --onset_threshold 0.8 --placebo_type pre_period
python src/01_revised_data_preparation.py --method effect_onset --vertical goed_nieuws --onset_threshold 0.8 --placebo_type pre_period

# === 3. Other Analyses (3 commands) ===
python src/01_revised_data_preparation.py --method artificial_stagger --vertical tech
python src/01_revised_data_preparation.py --method naive_launch
python src/01_revised_data_preparation.py --method thematic_tiers


Step 2: Render the Quarto Analysis Reports

# This command uses the parameter '80pct' to find the data from Step 1
quarto render src/02_analysis_revised.qmd --execute-param onset_pct:80pct
quarto render src/03_analysis_revised.qmd --execute-param onset_pct:80pct


To generate the outputs for the other runs, repeat this entire process, changing the threshold (e.g., 0.7) and parameter (e.g., 70pct) accordingly.