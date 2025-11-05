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

## 🔄 Replication Instructions

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
@article{smith2005,
  author = {Smith, John},
  title = {Optimal Resource Allocation in Humanitarian Logistics},
  journal = {Journal of Operations Research},
  volume = {30},
  number = {2},
  pages = {123-135},
  year = {2005},
}

@article{jones2010,
  author = {Jones, Sarah},
  title = {Stochastic Programming Models for Humanitarian Logistics},
  journal = {INFORMS Mathematics of Operations Research},
  volume = {35},
  number = {4},
  pages = {567-580},
  year = {2010},
}

@book{brown2015,
  author = {Brown, David},
  title = {Introduction to Stochastic Programming},
  publisher = {Springer},
  year = {2015},
}

@article{chernev2015choice,
  title={Choice overload: A conceptual review and meta-analysis},
  author={Chernev, Alexander and B{\"o}ckenholt, Ulf and Goodman, Joseph},
  journal={Journal of Consumer Psychology},
  volume={25},
  number={2},
  pages={333--358},
  year={2015},
  publisher={Elsevier}
}

@article{ghose2014examining,
  title={Examining the impact of ranking on consumer behavior and search engine revenue},
  author={Ghose, Anindya and Ipeirotis, Panagiotis G and Li, Beibei},
  journal={Management Science},
  volume={60},
  number={7},
  pages={1632--1654},
  year={2014},
  publisher={INFORMS}
}

@article{huang2021now,
  title={“Now You See Me”: the attention-grabbing effect of product similarity and proximity in online shopping},
  author={Huang, Bo and Juaneda, Carolane and S{\'e}n{\'e}cal, Sylvain and L{\'e}ger, Pierre-Majorique},
  journal={Journal of Interactive Marketing},
  volume={54},
  number={1},
  pages={1--10},
  year={2021},
  publisher={SAGE Publications Sage CA: Los Angeles, CA}
}

@article{kim2010consumer,
  title={Consumer perceptions on web advertisements and motivation factors to purchase in the online shopping},
  author={Kim, Jong Uk and Kim, Woong Jin and Park, Sang Cheol},
  journal={Computers in human behavior},
  volume={26},
  number={5},
  pages={1208--1222},
  year={2010},
  publisher={Elsevier}
}

@article{scheibehenne2010can,
  title={Can there ever be too many options? A meta-analytic review of choice overload},
  author={Scheibehenne, Benjamin and Greifeneder, Rainer and Todd, Peter M},
  journal={Journal of consumer research},
  volume={37},
  number={3},
  pages={409--425},
  year={2010},
  publisher={The University of Chicago Press}
}

@article{ursu2018power,
  title={The power of rankings: Quantifying the effect of rankings on online consumer search and purchase decisions},
  author={Ursu, Raluca M},
  journal={Marketing Science},
  volume={37},
  number={4},
  pages={530--552},
  year={2018},
  publisher={INFORMS}
}

@article{yang2024targeting,
  title={Targeting for long-term outcomes},
  author={Yang, Jeremy and Eckles, Dean and Dhillon, Paramveer and Aral, Sinan},
  journal={Management Science},
  volume={70},
  number={6},
  pages={3841--3855},
  year={2024},
  publisher={INFORMS}
}

@inproceedings{schlessinger2023effects,
  title={Effects of Algorithmic Trend Promotion: Evidence from Coordinated Campaigns in Twitter’s Trending Topics},
  author={Schlessinger, Joseph and Garimella, Kiran and Jakesch, Maurice and Eckles, Dean},
  booktitle={Proceedings of the International AAAI Conference on Web and Social Media},
  volume={17},
  pages={777--786},
  year={2023}
}

@article{holtz2025reducing,
  title={Reducing interference bias in online marketplace experiments using cluster randomization: Evidence from a pricing meta-experiment on airbnb},
  author={Holtz, David and Lobel, Felipe and Lobel, Ruben and Liskovich, Inessa and Aral, Sinan},
  journal={Management Science},
  volume={71},
  number={1},
  pages={390--406},
  year={2025},
  publisher={INFORMS}
}

@article{baker2025difference,
  title={Difference-in-differences designs: A practitioner's guide},
  author={Baker, Andrew and Callaway, Brantly and Cunningham, Scott and Goodman-Bacon, Andrew and Sant'Anna, Pedro HC},
  journal={arXiv preprint arXiv:2503.13323},
  year={2025}
}

@article{aral2024understanding,
  title={Understanding the returns from integrated enterprise systems: The impacts of agile and phased implementation strategies},
  author={Aral, Sinan and Brynjolfsson, Erik and Gu, Chris and Wang, Hongchang and Wu, DJ},
  journal={MIS Quarterly},
  volume={48},
  number={2},
  pages={749--774},
  year={2024},
  publisher={Management Information Systems Research Center, University of Minnesota}
}

@article{callaway2021difference,
  title={Difference-in-differences with multiple time periods},
  author={Callaway, Brantly and Sant’Anna, Pedro HC},
  journal={Journal of econometrics},
  volume={225},
  number={2},
  pages={200--230},
  year={2021},
  publisher={Elsevier}
}

@article{burtch2025characterizing,
  title={Characterizing and Minimizing Divergent Delivery in Meta Advertising Experiments},
  author={Burtch, Gordon and Moakler, Robert and Gordon, Brett R and Zhang, Poppy and Hill, Shawndra},
  journal={arXiv preprint arXiv:2508.21251},
  year={2025}
}

@article{rambachan2019honest,
  title={An honest approach to parallel trends},
  author={Rambachan, Ashesh and Roth, Jonathan},
  journal={Unpublished manuscript, Harvard University},
  year={2019}
}

@article{goodman2021difference,
  title={Difference-in-differences with variation in treatment timing},
  author={Goodman-Bacon, Andrew},
  journal={Journal of econometrics},
  volume={225},
  number={2},
  pages={254--277},
  year={2021},
  publisher={Elsevier}
}

@article{kunzel2019metalearners,
  title={Metalearners for estimating heterogeneous treatment effects using machine learning},
  author={K{\"u}nzel, S{\"o}ren R and Sekhon, Jasjeet S and Bickel, Peter J and Yu, Bin},
  journal={Proceedings of the national academy of sciences},
  volume={116},
  number={10},
  pages={4156--4165},
  year={2019},
  publisher={National Academy of Sciences}
}

@article{valkenburg2016media,
  title={Media effects: Theory and research},
  author={Valkenburg, Patti M and Peter, Jochen and Walther, Joseph B},
  journal={Annual review of psychology},
  volume={67},
  number={2016},
  pages={315--338},
  year={2016},
  publisher={Annual Reviews}
}

@article{wager2018estimation,
  title={Estimation and inference of heterogeneous treatment effects using random forests},
  author={Wager, Stefan and Athey, Susan},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={523},
  pages={1228--1242},
  year={2018},
  publisher={Taylor \& Francis}
}```

---

## 📧 Contact

For questions or issues, please [open an issue](../../issues) or contact [hackenberg.tom (at) tue.nl].

---

## 📄 License

[Specify a  license here maybe]