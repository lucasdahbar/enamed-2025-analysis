# ENAMED 2025 — Institutional Performance Prediction & Audit Hub

This repository investigates the predictive capacity of socioeconomic and institutional variables from the **ENAMED 2025** microdata to estimate course-level academic performance using machine learning models. 

This project accompanies two companion papers currently under review: an Educational Data Mining study, and a software-architecture study on institutional decision support.

📄 Read this README in Portuguese: README.pt-br.md

## Objective
Evaluate the predictive potential of institutional and socioeconomic aggregated variables from ENAMED microdata, with a specific focus on the **Federal University of Juiz de Fora (UFJF)**.

## Research Questions
- Which socioeconomic factors most significantly impact the average performance of medical courses?
- Can machine learning models accurately predict course-level performance categories based on institutional profiles?
- How does the performance of UFJF campuses compare to national and regional trends?

## Dataset & Data Source
ENAMED 2025 microdata provided by INEP.
**Note on Methodology:** Due to LGPD (Data Protection Law) compliance, the dataset does not allow individual student-level merging. Therefore, this study employs an **aggregate-level approach** by grouping data by Course Codes (`CO_CURSO`). Raw data is not included in this repository.

* **Official INEP Repository:** Link to the government website with the ENAMED 2025 microdata: [INEP Data Open Repository](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enamed)

## Project Pipeline & Core Architecture

The workflow is divided into 6 sequential core notebooks and an advanced statistical/generative expansion layer:

### Core Pipeline (Notebooks)
* **01_preprocessing.ipynb:** Raw ENAMED microdata ingestion, LGPD compliance aggregation by `CO_CURSO`, and performance metric calculations.
* **02_feature_selection_and_regression.ipynb:** Preliminary analysis exploring the impact of aggregated variables on general scores using regression models.
* **03_exam_perception_classification.ipynb:** Initial investigation into student exam perception columns as predictors for performance categorization.
* **04_feature_importance_and_multivariable_classification.ipynb:** Global feature importance mapping utilizing baseline Random Forest models to filter socioeconomic and institutional indicators.
* **05_feature_selection_and_model_optimization.ipynb:** Automatic noise filtering (dropping from 60 to 20 key features) and hyperparameter tuning using `GridSearchCV`. Achieved a **78% cross-validated accuracy** with optimized settings (`n_estimators: 50`, `max_depth: 6`, `criterion: 'entropy'`).
* **06_ufjf_insights_and_local_contributions.ipynb:** Deconstruction of the black-box model using Local Attribute Contribution Analysis to extract strategic, campus-specific pedagogical diagnostics for UFJF (Juiz de Fora and Governador Valadares).

### Advanced Statistical Validation (`analysis/`)
* **analysis_extras.py:** Robust statistics engine running comparison of 5 classifiers under stratified 5-fold cross-validation; a one-sided binomial significance test against chance level (+ Wilson 95% CI); mapping of the *complexity paradox curve* (accuracy vs. number of features); and data boundary error analysis.

## Key Findings & Regional Impact
* **UFJF Governador Valadares:** Classified as *High Performance* with a solid **94.6% prediction confidence**. Driven heavily by strong content retention (`I7_D`) and direct course appreciation (`I9_A`).
* **UFJF Juiz de Fora:** Correctly identified as *High Performance* but stabilizes as a boundary case (**57.6% confidence**) due to mixed internal student sentiment—balancing high praise for exam clarity (`I4_B`) with localized friction caused by perceived content difficulty (`I6_A` and `I1_D`).

## Interactive Audit Dashboard & LLM Layer

To translate the pipeline's mathematical coefficients into actionable insights for educational managers, an interactive Web Dashboard was developed using **Streamlit**. 

### Key Dashboard Features:
* **Dynamic Search Framework:** Instantly look up any medical program in Brazil by filtering its unique Course Code (`CO_CURSO`).
* **Live Feature Contribution Mapping:** Generates horizontal bar charts on the fly showing the exact local impact force scores derived from the optimized Random Forest model, re-sorting features dynamically per query.
* **Automated Pedagogical Diagnostics:** Programmatically parses student survey responses to highlight localized strengths and critical curricular friction points.
* **Grounded LLM Explanation Layer (`llm_explainer.py`):** An optional, safe natural-language explanation layer powered by Gemini (`gemini-3.5-flash`). It processes ONLY pre-computed numbers to generate text summaries for coordinators, ensuring full data privacy with a rule-based fallback if the service is offline.
* **Native Bilingual Toggle:** Full UI, charts, and data pack translations between **English** and **Português** via a sidebar controller.

---

## Getting Started & How to Run

### 1. Environment Setup
Clone this repository and ensure you have the required stacks installed:

```bash
git clone [https://github.com/lucasdahbar/enamed-performance-prediction.git](https://github.com/lucasdahbar/enamed-performance-prediction.git)
cd enamed-performance-prediction
pip install pandas numpy scikit-learn scipy matplotlib google-genai streamlit
```
### 2. Running the Statistical Analyses
Execute the validation script against your raw data path to print benchmarks and export the metrics (results/results.json and results/accuracy_vs_k.png):

```bash
python analysis/analysis_extras.py --data data/raw/microdados_enade_2025_arq3.txt --outdir results --scan-seeds
```

### 3. Launching the Dashboard (With LLM Support)
To enable the generative explanation tab, generate an API key at Google AI Studio and place it inside .streamlit/secrets.toml:

```Ini, TOML
GEMINI_API_KEY = "your-actual-api-key-here"
```
Then launch the Streamlit server using the explicit Python module flag:

```Bash
python -m streamlit run app.py
```