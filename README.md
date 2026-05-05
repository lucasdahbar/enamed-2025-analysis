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
**Note on Methodology:** Due to LGPD (Data Protection Law) compliance, the dataset does not allow individual student-level merging. Therefore, this study employs an **aggregate-level approach** by grouping data by Course Codes CO_CURSO)

Raw data is not included in this repository.

## Methodology
- Data filtering (Focusing on UFJF codes: 13103 and 5001167)
- Data preprocessing and course-level feature aggregation
- Supervised learning (classification and regression)
- Baseline models: Random Forest and XGBoost
- Evaluation metrics: Accuracy, R-squared and F1-score

## Project Status
Initial setup and data filtering.

## Related Work
This project builds upon previous experiences with ENADE microdata analysis, transitioning the focus to the newly established medical examination (ENAMED) and its specific evaluation matrix (Internal Medicine, Surgery, Pediatrics, etc.).

References:
https://github.com/Ivanylson/Ontology_ENADE
https://github.com/lucasdahbar/enade-performance-prediction
