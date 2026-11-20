import os
import time
import urllib.request
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"
DATA_FILE = "seeds_dataset.txt"
OUTPUT_DIR = "outputs"

FEATURES = [
    "area", "perimeter", "compactness", "kernel_length",
    "kernel_width", "asymmetry_coefficient", "kernel_groove_length"
]
VARIETY_NAMES = {1: "Kama", 2: "Rosa", 3: "Canadian"}


class exc(Exception):
    """Custom exception for dataset/download issues and validation errors."""

    def __init__(self, message="An error occurred.", details=None):
        super().__init__(message)
        self.message = message
        self.details = details

    def __str__(self):
        if self.details:
            return f"{self.message}: {self.details}"
        
        return self.message

    def __repr__(self):
        return f"exc(message={self.message!r}, details={self.details!r})"

    def to_dict(self):
        return {"message": self.message, "details": self.details}


def load_dataset():
    if not os.path.exists(DATA_FILE):
        print("Dataset not found. Downloading from the UCI repository...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        except Exception as exc_error:
            raise RuntimeError(
                "Could not download the dataset. Download seeds_dataset.txt "
                "from https://archive.ics.uci.edu/dataset/236/seed and place "
                "it in this project folder."
            ) from exc_error

    df = pd.read_csv(DATA_FILE, sep=r"\s+", header=None, names=FEATURES + ["variety"])
    df["variety_name"] = df["variety"].map(VARIETY_NAMES)
    return df


def analyze_chunk(args):
    model_name, model, X_train, X_test, y_train, y_test = args
    start = time.perf_counter()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    elapsed = time.perf_counter() - start
    return model_name, model, predictions, elapsed


def save_eda_plots(df):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="variety_name", order=["Kama", "Rosa", "Canadian"])
    plt.title("Number of Samples per Wheat Variety")
    plt.xlabel("Wheat variety")
    plt.ylabel("Number of samples")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(11, 8))
    sns.heatmap(df[FEATURES].corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Between Seed Features")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_correlation.png"), dpi=160)
    plt.close()

    pair = sns.pairplot(df, vars=FEATURES[:5], hue="variety_name", corner=True, diag_kind="hist")
    pair.fig.suptitle("Seed Feature Distributions by Variety", y=1.02)
    pair.savefig(os.path.join(OUTPUT_DIR, "feature_pairplot.png"), dpi=160)
    plt.close("all")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset()

    print("\n=== DATASET OVERVIEW ===")
    print("Shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head().to_string(index=False))
    print("\nFeature data types:")
    print(df[FEATURES].dtypes)
    print("\nMissing values:")
    print(df[FEATURES + ["variety"]].isnull().sum())
    print("\nClass counts:")
    print(df["variety_name"].value_counts().to_string())
    print("\nDescriptive statistics:")
    print(df[FEATURES].describe().round(3).to_string())

    df.to_csv(os.path.join(OUTPUT_DIR, "dataset_with_labels.csv"), index=False)
    save_eda_plots(df)

    X = df[FEATURES]
    y = df["variety"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4, random_state=42, class_weight=None
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=42))
        ])
    }

    # Parallel model evaluation: each worker trains/evaluates one model.
    jobs = [
        (name, model, X_train, X_test, y_train, y_test)
        for name, model in models.items()
    ]
    start_parallel = time.perf_counter()
    with Pool(processes=min(len(jobs), cpu_count())) as pool:
        results = pool.map(analyze_chunk, jobs)
    parallel_wall_time = time.perf_counter() - start_parallel

    summary = []
    prediction_frame = pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "actual_variety": y_test.map(VARIETY_NAMES).to_numpy()
    }, index=y_test.index)

    fitted_models = {}
    for name, model, pred, elapsed in results:
        fitted_models[name] = model
        acc = accuracy_score(y_test, pred)
        summary.append({
            "model": name,
            "accuracy": acc,
            "training_and_prediction_seconds": elapsed
        })
        prediction_frame[name + "_prediction"] = pred
        prediction_frame[name + "_variety"] = pd.Series(pred, index=y_test.index).map(VARIETY_NAMES)
        print(f"\n=== {name.upper()} ===")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(
            y_test, pred, labels=[1, 2, 3],
            target_names=["Kama", "Rosa", "Canadian"], zero_division=0
        ))
        cm = confusion_matrix(y_test, pred, labels=[1, 2, 3])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Kama", "Rosa", "Canadian"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        safe_name = name.lower().replace(" ", "_")
        plt.savefig(os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png"), dpi=160)
        plt.close()

    summary_df = pd.DataFrame(summary).sort_values("accuracy", ascending=False)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison.csv"), index=False)
    prediction_frame.sort_index().to_csv(os.path.join(OUTPUT_DIR, "prediction_comparison.csv"), index_label="row_index")

    print("\n=== MODEL COMPARISON ===")
    print(summary_df.to_string(index=False))
    print(f"\nParallel evaluation wall time: {parallel_wall_time:.4f} seconds")

    # Feature importance: tree impurity importance and permutation importance.
    tree = fitted_models["Decision Tree"]
    tree_importance = pd.Series(tree.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    tree_importance.plot(kind="barh")
    plt.title("Decision Tree Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "decision_tree_feature_importance.png"), dpi=160)
    plt.close()

    logreg = fitted_models["Logistic Regression"]
    perm = permutation_importance(
        logreg, X_test, y_test, n_repeats=20, random_state=42, scoring="accuracy"
    )
    perm_importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values()
    plt.figure(figsize=(8, 5))
    perm_importance.plot(kind="barh")
    plt.title("Logistic Regression Permutation Importance")
    plt.xlabel("Mean decrease in accuracy after permutation")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "logistic_regression_permutation_importance.png"), dpi=160)
    plt.close()

    # PCA visualisation: scale all features, project to two dimensions.
    scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(scaled)
    pca_df = pd.DataFrame({
        "PC1": components[:, 0],
        "PC2": components[:, 1],
        "Variety": df["variety_name"]
    })
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Variety", s=65)
    plt.title("Wheat Varieties Visualised with PCA")
    plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pca_variety_visualization.png"), dpi=160)
    plt.close()

    # Save a text report with reproducible run-specific results.
    with open(os.path.join(OUTPUT_DIR, "run_summary.txt"), "w", encoding="utf-8") as f:
        f.write("WHEAT SEED VARIETY CLASSIFICATION - RUN SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Rows: {len(df)}\n")
        f.write(f"Features: {len(FEATURES)}\n")
        f.write(f"Train samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n\n")
        f.write(summary_df.to_string(index=False))
        f.write(f"\n\nParallel evaluation wall time: {parallel_wall_time:.4f} seconds\n\n")
        f.write("Decision Tree feature importance:\n")
        f.write(tree_importance.sort_values(ascending=False).to_string())
        f.write("\n\nLogistic Regression permutation importance:\n")
        f.write(perm_importance.sort_values(ascending=False).to_string())
        f.write("\n")

    print("\n=== FEATURE IMPORTANCE: DECISION TREE ===")
    print(tree_importance.sort_values(ascending=False).round(4).to_string())
    print("\n=== FEATURE IMPORTANCE: LOGISTIC REGRESSION PERMUTATION ===")
    print(perm_importance.sort_values(ascending=False).round(4).to_string())
    print("\nDone. Charts and result files are saved in the 'outputs' folder.")


if __name__ == "__main__":
    main()
