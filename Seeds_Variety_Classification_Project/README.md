# Wheat Seed Variety Classification Using Machine Learning

## Overview
This project predicts wheat variety (Kama, Rosa, or Canadian) from seven numerical seed measurements. It compares a Decision Tree and Logistic Regression, generates evaluation metrics and visualisations, and runs the two model evaluations concurrently using Python multiprocessing.

**Important:** multiprocessing on one machine is a parallel-processing demonstration, not a multi-machine cluster. For a true cluster deployment, workers would run on separate networked nodes using a framework such as Apache Spark.

## Dataset
UCI Machine Learning Repository — Seeds dataset:
https://archive.ics.uci.edu/dataset/236/seed

- 210 observations
- 7 numeric geometric features
- 3 varieties, 70 observations per variety
- No missing values reported by UCI
- Features: area, perimeter, compactness, kernel length, kernel width, asymmetry coefficient, kernel groove length
- Labels: 1 = Kama, 2 = Rosa, 3 = Canadian

The script automatically downloads the original whitespace-separated file. If download fails, download `seeds_dataset.txt` from the UCI dataset page and place it beside `main.py`.

## Requirements
Python 3.10+ recommended.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run:
```bash
python main.py
```

## Outputs
The `outputs/` directory will contain:
- `class_distribution.png`
- `feature_correlation.png`
- `feature_pairplot.png`
- confusion matrices for both classifiers
- `decision_tree_feature_importance.png`
- `logistic_regression_permutation_importance.png`
- `pca_variety_visualization.png`
- `model_comparison.csv`
- `prediction_comparison.csv`
- `dataset_with_labels.csv`
- `run_summary.txt`

Results depend on the test split, library versions, and model settings. Use the generated outputs in the final report; do not invent accuracy values.

## Method
1. Load the UCI dataset and inspect rows, types, missing values, class counts and descriptive statistics.
2. Use a stratified 75/25 train/test split with fixed random seed 42.
3. Apply StandardScaler inside the Logistic Regression pipeline. The scaler is fitted only on training data through the pipeline to avoid test-data leakage.
4. Train a Decision Tree (maximum depth 4) and Logistic Regression (maximum 3000 iterations).
5. Report accuracy, precision, recall, F1-score, and confusion matrices.
6. Compare predictions and assess feature importance.
7. Execute model evaluations concurrently using multiprocessing and record run times. Since the models are small, parallel execution may not be faster due to process startup overhead.

## Academic note
This dataset and project are suitable for supervised classification. The PCA plot is a two-dimensional projection for visual exploration; it is not the classifier itself.
