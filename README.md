# ENAMED 2025 Institutional Performance Prediction

This project investigates the predictive capacity of socioeconomic and institutional variables from the **ENAMED 2025** microdata to estimate course-level academic performance using machine learning models.

The ENAMED is a specialized modality of the national exam (ENADE) specifically designed for medical schools in Brazil. This research explores the dataset to identify patterns that influence medical education quality.

📄 Read this README in Portuguese: README.pt-br.md

## Objective
Evaluate the predictive potential of institutional and socioeconomic aggregated variables from ENAMED microdata, with a specific focus on the **Federal University of Juiz de Fora (UFJF)**.

## Research Questions
- Which socioeconomic factors most significantly impact the average performance of medical courses?
- Can machine learning models accurately predict course-level performance categories based on institutional profiles?
- How does the performance of UFJF campuses compare to national and regional trends?

## Dataset
ENAMED 2025 microdata provided by INEP.
**Note on Methodology:** Due to LGPD (Data Protection Law) compliance, the dataset does not allow individual student-level merging. Therefore, this study employs an **aggregate-level approach** by grouping data by Course Codes (CO_CURSO).

Raw data is not included in this repository.

## Project Pipeline & Architecture

The workflow is divided into 6 sequential notebooks structured as follows:

* **01_preprocessing.ipynb:** Raw ENAMED microdata ingestion, LGPD compliance aggregation by `CO_CURSO`, and performance metric calculations.
* **02_feature_selection_and_regression.ipynb:** Preliminary analysis exploring the impact of aggregated variables on general scores using regression models.
* **03_exam_perception_classification.ipynb:** Initial investigation into student exam perception columns as predictors for performance categorization.
* **04_feature_importance_and_multivariable_classification.ipynb:** Global feature importance mapping utilizing baseline Random Forest models to filter socioeconomic and institutional indicators.
* **05_feature_selection_and_model_optimization.ipynb:** Automatic noise filtering (dropping from 60 to 20 key features) and hyperparameter tuning using `GridSearchCV`. Achieved a **78% cross-validated accuracy** with optimized settings (`n_estimators: 50`, `max_depth: 6`, `criterion: 'entropy'`).
* **06_ufjf_insights_and_local_contributions.ipynb:** Deconstruction of the black-box model using Local Attribute Contribution Analysis to extract strategic, campus-specific pedagogical diagnostics for UFJF (Juiz de Fora and Governador Valadares).

## Project Status
**Completed.** The machine learning pipeline is fully established, and institutional coefficients have been successfully translated into actionable management insights.

## Key Findings & Regional Impact
* **UFJF Governador Valadares:** Classified as *High Performance* with a solid **94.6% prediction confidence**. Driven heavily by strong content retention (`I7_D`) and direct course appreciation (`I9_A`).
* **UFJF Juiz de Fora:** Correctly identified as *High Performance* but stabilizes as a boundary case (**57.6% confidence**) due to mixed internal student sentiment—balancing high praise for exam clarity (`I4_B`) with localized friction caused by perceived content difficulty (`I6_A` and `I1_D`).

## Interactive Audit Dashboard

To translate the pipeline's mathematical coefficients into actionable insights for educational managers, an interactive Web Dashboard was developed using **Streamlit**. The interface functions as an institutional auditing platform tailored for pedagogical decision-making.

### Key Dashboard Features:
* **Dynamic Search Framework:** Instantly look up any medical program in Brazil by filtering its unique Course Code (`CO_CURSO`).
* **Live Feature Contribution Mapping:** Generates horizontal bar charts on the fly showing the exact local impact force scores derived from the optimized Random Forest model, re-sorting features dynamically per query.
* **Automated Pedagogical Diagnostics:** Programmatically parses student survey responses to highlight localized strengths and critical curricular friction points.
* **Native Bilingual Toggle:** Full UI, charts, and data pack translations between **English** and **Português** via a sidebar controller.
* **LGPD-by-Design Compliance:** Built entirely on masked, aggregate identifiers to preserve data privacy and prevent individual re-identification.

## Related Work
This project builds upon previous experiences with ENADE microdata analysis, transitioning the focus to the newly established medical examination (ENAMED) and its specific evaluation matrix.

References:
[https://github.com/Ivanylson/Ontology_ENADE](https://github.com/Ivanylson/Ontology_ENADE)

[https://github.com/lucasdahbar/enade-performance-prediction](https://github.com/lucasdahbar/enade-performance-prediction)

Link to the gov website with the ENAMED 2025 microdata:

[https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enamed](https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enamed)