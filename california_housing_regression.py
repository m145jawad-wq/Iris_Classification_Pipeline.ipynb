# Real Estate Housing Price Prediction (Ridge & Lasso Regression)
# Install: pip install numpy pandas matplotlib seaborn scikit-learn
# Run: python california_housing_regression.py

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
OUTPUT_DIR = "california_housing_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid")

# 1. Load dataset (first run may need internet access)
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
X = housing.data.copy()
y = housing.target.copy()  # Target units are $100,000.
print("Dataset shape:", X.shape)
print(X.head())
print("\nMissing values:")
print(X.isna().sum())

# 2. Correlation heatmap; median imputation used for EDA only
eda_imputer = SimpleImputer(strategy="median")
eda_X = pd.DataFrame(eda_imputer.fit_transform(X), columns=X.columns, index=X.index)
corr_df = eda_X.copy()
corr_df["MedHouseVal"] = y
plt.figure(figsize=(11, 8))
sns.heatmap(corr_df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("California Housing Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_correlation_heatmap.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

# 3. 80/20 split before fitting preprocessing steps
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE
)
print("Training rows:", len(X_train), "| Testing rows:", len(X_test))

# 4. Preprocessing: impute missing values, create pairwise interaction features, scale
preprocess = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
    ("scaler", StandardScaler())
])

# 5. OLS, Ridge and Lasso; tune regularization strength with 5-fold GridSearchCV
model_specs = {
    "OLS Linear Regression": (
        Pipeline([("preprocess", preprocess), ("model", LinearRegression())]), None
    ),
    "Ridge Regression": (
        Pipeline([("preprocess", preprocess), ("model", Ridge())]),
        {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
    ),
    "Lasso Regression": (
        Pipeline([("preprocess", preprocess), ("model", Lasso(max_iter=20000, random_state=RANDOM_STATE))]),
        {"model__alpha": [0.0001, 0.001, 0.01, 0.1, 1.0]}
    )
}

results = []
predictions = {}
fitted_models = {}

for name, (pipeline, grid) in model_specs.items():
    print("\n" + "=" * 60)
    print("Training:", name)
    if grid is not None:
        search = GridSearchCV(
            pipeline, grid, scoring="neg_root_mean_squared_error",
            cv=5, n_jobs=-1
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        print("Best parameters:", search.best_params_)
        print("Best CV RMSE:", round(-search.best_score_, 4))
    else:
        model = pipeline.fit(X_train, y_train)

    pred = model.predict(X_test)
    fitted_models[name] = model
    predictions[name] = pred

    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE:  {mae:.4f}")
    print(f"Test R2:   {r2:.4f}")

# 6. Save model metrics table and comparison chart
metrics = pd.DataFrame(results).set_index("Model")
print("\nFinal test-set comparison:")
print(metrics.round(4))
metrics.to_csv(f"{OUTPUT_DIR}/model_metrics.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for metric, ax in zip(["RMSE", "MAE", "R2"], axes):
    metrics[metric].plot(kind="bar", ax=ax, rot=20, legend=False)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.grid(axis="y", alpha=0.3)
fig.suptitle("OLS vs Ridge vs Lasso: Test Metrics")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_model_metrics_comparison.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

# 7. Actual vs predicted plots
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for (name, pred), ax in zip(predictions.items(), axes):
    ax.scatter(y_test, pred, alpha=0.25, s=12)
    low = min(float(y_test.min()), float(np.min(pred)))
    high = max(float(y_test.max()), float(np.max(pred)))
    ax.plot([low, high], [low, high], linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Actual median house value")
    ax.set_ylabel("Predicted median house value")
plt.suptitle("Actual vs Predicted Values")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_actual_vs_predicted.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

# 8. Residual distributions
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for (name, pred), ax in zip(predictions.items(), axes):
    residuals = np.asarray(y_test) - np.asarray(pred)
    sns.histplot(residuals, kde=True, bins=40, ax=ax)
    ax.axvline(0, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Residual (actual - predicted)")
    ax.set_ylabel("Count")
plt.suptitle("Residual Error Distributions")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_residual_distributions.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

# 9. Residuals vs predicted values
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for (name, pred), ax in zip(predictions.items(), axes):
    residuals = np.asarray(y_test) - np.asarray(pred)
    ax.scatter(pred, residuals, alpha=0.25, s=12)
    ax.axhline(0, linestyle="--")
    ax.set_title(name)
    ax.set_xlabel("Predicted median house value")
    ax.set_ylabel("Residual")
plt.suptitle("Residuals vs Predicted Values")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_residuals_vs_predicted.png", dpi=160, bbox_inches="tight")
plt.show()
plt.close()

print("\nProject completed.")
print("Graphs and model_metrics.csv are saved in:", OUTPUT_DIR)
print("Target values are in units of $100,000 as defined by scikit-learn.")
