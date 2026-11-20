1# Wheat Seed Variety Classification Using Machine Learning and Parallel Model Evaluation

## Abstract
This project studies the classification of wheat seed varieties using seven numerical measurements of seed geometry. The UCI Seeds dataset contains 210 observations from three wheat varieties: Kama, Rosa and Canadian. A Decision Tree classifier and a Logistic Regression classifier are trained and evaluated using a stratified train/test split. The work includes exploratory analysis, feature standardisation, per-class metrics, confusion matrices, prediction comparison, feature-importance analysis and visualisation using Principal Component Analysis (PCA). A multiprocessing experiment runs model evaluations concurrently to demonstrate parallel execution. Actual metric values must be inserted from the generated output files after running the program.

## 1. Introduction
Seed classification is useful in agricultural quality assessment, grain sorting and variety identification. Seed geometry provides measurable signals that can be analysed computationally. Machine learning can learn relationships between these measurements and known variety labels, enabling the prediction of a variety for a previously unseen seed.

Cluster computing distributes computation across multiple processing nodes. In this implementation, Python multiprocessing is used to run two model-evaluation jobs concurrently on one computer. This is a small-scale parallel-processing demonstration; it should not be described as a physical multi-machine cluster.

## 2. Problem Statement
Manual identification of wheat varieties can be time-consuming and inconsistent. The project investigates whether seven numerical measurements can reliably classify a wheat seed as Kama, Rosa or Canadian, and compares a tree-based model with a linear classifier.

## 3. Objectives
1. Load and inspect seven numerical features and the variety label.
2. Examine dataset structure, class balance, missing values and feature distributions.
3. Scale features for Logistic Regression without leaking test-set information.
4. Train Decision Tree and Logistic Regression classifiers.
5. Evaluate accuracy, precision, recall, F1-score and confusion matrices.
6. Compare predictions and identify informative features.
7. Visualise feature relationships and class separation.
8. Demonstrate concurrent model evaluation and record execution time.

## 4. Dataset Description
Source: UCI Machine Learning Repository, Seeds dataset, DOI: 10.24432/C5H30K. The dataset contains 210 observations, with 70 examples for each of three wheat varieties. UCI reports seven continuous geometric attributes and no missing values.

| Feature | Description |
|---|---|
| Area | Area of the seed kernel |
| Perimeter | Perimeter of the seed kernel |
| Compactness | Compactness measure, defined as 4πA/P² |
| Kernel length | Length of the kernel |
| Kernel width | Width of the kernel |
| Asymmetry coefficient | Measurement of kernel asymmetry |
| Kernel groove length | Length of the kernel groove |
| Variety | Target class: 1 Kama, 2 Rosa, 3 Canadian |

## 5. Methodology
### 5.1 Data loading and inspection
The script loads the whitespace-separated dataset, assigns descriptive column names, maps numeric class codes to variety names and reports data shape, sample rows, types, missing values, class counts and descriptive statistics.

### 5.2 Data preparation
The seven measurements are used as predictor variables and variety is the target. A stratified 75%/25% train/test split with random state 42 keeps the class proportions approximately consistent. StandardScaler is used inside a scikit-learn Pipeline for Logistic Regression. This ensures the scaler is fitted on training data only. Decision Trees do not require feature scaling, but they use the same training and testing observations for a fair comparison.

### 5.3 Models
**Decision Tree:** learns decision rules by splitting feature values. The maximum depth is limited to 4 to reduce unnecessary complexity.

**Logistic Regression:** estimates class probabilities using a linear decision function. Standardised features help the optimisation process because the features have different numeric scales.

### 5.4 Evaluation metrics
- **Accuracy:** fraction of all test predictions that are correct.
- **Precision:** of the samples predicted as a class, the fraction that truly belong to that class.
- **Recall:** of the actual samples in a class, the fraction correctly identified.
- **F1-score:** harmonic mean of precision and recall.
- **Confusion matrix:** shows actual classes against predicted classes.

### 5.5 Feature analysis
Decision Tree feature importance measures the contribution of features to the tree's impurity reduction. Permutation importance for Logistic Regression measures how much test accuracy falls when one feature is shuffled. These measures are model-dependent and should not be interpreted as proof of causation.

### 5.6 Visualisation
The program creates a class-distribution chart, correlation heatmap, feature pairplot, confusion matrices, feature-importance charts and a PCA scatter plot. PCA projects the standardised seven-dimensional data into two dimensions for visual inspection; it is not used as a replacement for the classifiers.

### 5.7 Parallel execution
Python multiprocessing executes the two model-evaluation jobs concurrently using separate worker processes. Wall-clock time is recorded. Because the dataset and models are small, process startup overhead may mean parallel execution is not faster than sequential execution. A true cluster extension could use Apache Spark MLlib on multiple machines.

## 6. Results and Analysis
Run `python main.py` and transfer the actual results into the tables below.

### 6.1 Dataset inspection
Report the dataset shape, missing-value count, class distribution and descriptive statistics printed by the program. Expected dataset metadata from UCI: 210 rows, seven features and three classes with 70 samples each.

### 6.2 Model comparison
Fill this table from `outputs/model_comparison.csv` and the console classification reports.

| Metric | Decision Tree | Logistic Regression |
|---|---:|---:|
| Accuracy | Insert actual result | Insert actual result |
| Macro precision | Insert actual result | Insert actual result |
| Macro recall | Insert actual result | Insert actual result |
| Macro F1-score | Insert actual result | Insert actual result |
| Evaluation time (seconds) | Insert actual result | Insert actual result |

### 6.3 Per-class performance
Copy precision, recall, F1-score and support for Kama, Rosa and Canadian from the classification reports. Do not assume one model is superior before inspecting the actual outputs.

### 6.4 Confusion matrices
Review `confusion_matrix_decision_tree.png` and `confusion_matrix_logistic_regression.png`. Diagonal cells are correct predictions. Off-diagonal cells indicate which varieties are confused with one another.

### 6.5 Feature importance and variety differences
Use the two feature-importance charts and the pairplot. Discuss which features rank highly for each model and whether varieties appear separated by area, perimeter, compactness, length, width, asymmetry or groove length. The UCI dataset provides measured geometric properties, but specific claims about the most important features should be based on the generated results rather than assumed in advance.

### 6.6 Parallel-processing analysis
Record the parallel wall-clock time from `outputs/run_summary.txt`. Explain that concurrent model jobs illustrate parallel processing. With only 210 rows, timing is highly affected by process creation and system load, so this is not evidence of cluster speedup. A rigorous cluster experiment would repeat the experiment with larger workloads and multiple networked workers.

## 7. Conclusion
This project builds and compares two supervised classifiers for three wheat varieties using seven geometric features. The resulting accuracy and per-class metrics establish how well each model performs on the held-out test set. Confusion matrices identify misclassified varieties, while feature importance and PCA visualisations help explain the patterns in the measurements. Multiprocessing demonstrates concurrent evaluation, but a true distributed cluster implementation would require multiple nodes or a distributed framework.

## 8. Limitations and Future Scope
- The dataset is small and represents three varieties.
- Results may vary with the train/test split and model hyperparameters.
- Feature importance differs by model and is not causal evidence.
- Multiprocessing on one computer is not a multi-node cluster.
- Future work can use cross-validation, hyperparameter tuning, a larger grain dataset, and Apache Spark MLlib on a multi-node cluster.

## 9. References
1. Charytanowicz, M., Niewczas, J., Kulczycki, P., Kowalski, P., & Łukasik, S. (2010). *Seeds*. UCI Machine Learning Repository. https://doi.org/10.24432/C5H30K
2. UCI Machine Learning Repository. *Seeds dataset documentation*. https://archive.ics.uci.edu/dataset/236/seed
3. scikit-learn developers. *classification_report documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
4. scikit-learn developers. *confusion_matrix documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
5. scikit-learn developers. *StandardScaler documentation*. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
