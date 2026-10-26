# ============================================================
# Multi-Class Benchmark Classification Pipeline (Iris Dataset)
# Models: Logistic Regression and Decision Tree
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# -------------------- Configuration --------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
OUTPUT_DIR = "iris_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")

# -------------------- 1. Load Dataset --------------------
iris = load_iris(as_frame=True)
X = iris.data.copy()
y = iris.target.copy()
class_names = list(iris.target_names)

df = X.copy()
df["species"] = y.map(dict(enumerate(class_names)))

print("=" * 65)
print("IRIS MULTI-CLASS CLASSIFICATION PIPELINE")
print("=" * 65)
print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())
print("\nSpecies counts:")
print(df["species"].value_counts())

# -------------------- 2. Pairplot --------------------
print("\nCreating pairplot...")
pairplot = sns.pairplot(
    df,
    hue="species",
    diag_kind="hist",
    corner=True,
    plot_kws={"alpha": 0.8, "s": 35}
)
pairplot.fig.suptitle("Iris Dataset Pairplot by Species", y=1.02)
pairplot.fig.savefig(
    os.path.join(OUTPUT_DIR, "01_iris_pairplot.png"),
    dpi=160,
    bbox_inches="tight"
)
plt.show()
plt.close(pairplot.fig)

# -------------------- 3. Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# -------------------- 4. Define Models --------------------
# Scaling is applied to Logistic Regression using a Pipeline.
# Decision Tree does not require feature scaling.
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=4
    )
}

# -------------------- 5. Train and Evaluate --------------------
results = []
predictions = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[model_name] = y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision (Macro)": precision,
        "Recall (Macro)": recall,
        "F1-score (Macro)": f1
    })

    print("\n" + "=" * 65)
    print(model_name)
    print("=" * 65)
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro Precision:   {precision:.4f}")
    print(f"Macro Recall:      {recall:.4f}")
    print(f"Macro F1-score:    {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        zero_division=0
    ))

# -------------------- 6. Confusion Matrix Heatmaps --------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for (model_name, model), ax in zip(models.items(), axes):
    y_pred = predictions[model_name]
    cm = confusion_matrix(
        y_test, y_pred, labels=np.arange(len(class_names))
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax
    )
    ax.set_title(f"{model_name}\nConfusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "02_confusion_matrices.png"),
    dpi=160,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)

# -------------------- 7. Compare Model Metrics --------------------
results_df = pd.DataFrame(results).set_index("Model")

print("\n" + "=" * 65)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 65)
print(results_df.round(4))

results_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"))

ax = results_df.plot(
    kind="bar",
    figsize=(11, 6),
    ylim=(0, 1.05),
    rot=0
)
ax.set_title("Classification Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.legend(loc="lower right")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "03_metrics_comparison.png"),
    dpi=160,
    bbox_inches="tight"
)
plt.show()
plt.close()

# -------------------- 8. Decision Boundary Visualization --------------------
# A 2D graph can show only two features. We use petal length and width.
feature_pair = ["petal length (cm)", "petal width (cm)"]
X_2d = X[feature_pair]

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_2d,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

boundary_models = {
    "Logistic Regression (2 features)": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ))
    ]),
    "Decision Tree (2 features)": DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=4
    )
}

x_min = X_2d.iloc[:, 0].min() - 0.5
x_max = X_2d.iloc[:, 0].max() + 0.5
y_min = X_2d.iloc[:, 1].min() - 0.5
y_max = X_2d.iloc[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

grid = pd.DataFrame(
    np.c_[xx.ravel(), yy.ravel()],
    columns=feature_pair
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for (model_name, model), ax in zip(boundary_models.items(), axes):
    model.fit(X2_train, y2_train)
    grid_predictions = model.predict(grid)
    Z = np.asarray(grid_predictions).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.25, cmap="viridis")
    scatter = ax.scatter(
        X2_test.iloc[:, 0],
        X2_test.iloc[:, 1],
        c=y2_test,
        cmap="viridis",
        edgecolor="black",
        s=55
    )
    ax.set_title(model_name)
    ax.set_xlabel("Petal Length (cm)")
    ax.set_ylabel("Petal Width (cm)")
    ax.grid(alpha=0.2)

handles, _ = scatter.legend_elements()
fig.legend(
    handles,
    class_names,
    title="True Species",
    loc="lower center",
    ncol=3
)
fig.suptitle(
    "Decision Boundaries: Petal Length vs Petal Width",
    y=1.02
)
plt.tight_layout(rect=[0, 0.12, 1, 1])
plt.savefig(
    os.path.join(OUTPUT_DIR, "04_decision_boundaries.png"),
    dpi=160,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)

# -------------------- 9. Final Summary --------------------
print("\n" + "=" * 65)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 65)
print(f"All graphs and the metrics CSV have been saved in: {OUTPUT_DIR}")
print("Generated files:")
print("  01_iris_pairplot.png")
print("  02_confusion_matrices.png")
print("  03_metrics_comparison.png")
print("  04_decision_boundaries.png")
print("  model_metrics.csv")
