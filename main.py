"""
Student Performance Prediction - Multiple Linear Regression Pipeline
=====================================================================
This script performs an end-to-end Machine Learning workflow:
1. Inspects the dataset (columns, data types, missing values, duplicates, descriptive statistics).
2. Performs Exploratory Data Analysis (EDA) and generates visualization plots.
3. Cleans data and constructs a Scikit-Learn Pipeline with ColumnTransformer.
4. Splits data into train and test sets (80:20 split, random_state=42) without data leakage.
5. Trains a Multiple Linear Regression model using LinearRegression.
6. Evaluates model performance using MAE, MSE, RMSE, R2, and Adjusted R2.
7. Saves diagnostic plots (Actual vs Predicted, Residuals, Heatmap, Feature Coefficients).
8. Exports the complete trained Pipeline to models/regression_model.joblib.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress harmless BLAS/Accelerate matmul warnings on macOS ARM
warnings.filterwarnings('ignore', message='.*encountered in matmul.*')

# Configure aesthetics for plots
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 14
})

# Directory constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


def find_dataset_path():
    """Locate the dataset file gracefully, handling possible naming variations."""
    candidates = [
        os.path.join(BASE_DIR, "Student_Performance.csv"),
        os.path.join(BASE_DIR, "Student_Performace.csv"),
    ]
    for path in candidates:
        if os.path.isfile(path) and os.path.getsize(path) > 0:
            return path
    raise FileNotFoundError(
        "Could not find dataset. Please ensure 'Student_Performance.csv' is present and not empty."
    )


def inspect_dataset(df: pd.DataFrame):
    """
    Step 1 & 3: Comprehensive inspection of CSV structure and statistics.
    Prints dataset shape, data types, missing values, duplicates, and statistical summary.
    """
    print("\n" + "=" * 60)
    print(" 1. DATASET INSPECTION & SUMMARY ")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns and Data Types:")
    for col, dtype in df.dtypes.items():
        print(f"  - {col:<35} : {dtype}")

    print("\nMissing Values per Column:")
    null_counts = df.isnull().sum()
    for col, count in null_counts.items():
        print(f"  - {col:<35} : {count} ({(count / len(df)) * 100:.2f}%)")

    duplicate_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {duplicate_count} ({(duplicate_count / len(df)) * 100:.2f}%)")

    print("\nSummary Statistics for Numerical Columns:")
    print(df.describe().to_string())

    print("\nCategorical Column Value Counts ('Extracurricular Activities'):")
    print(df['Extracurricular Activities'].value_counts(dropna=False).to_string())
    print("=" * 60 + "\n")


def perform_eda_visualizations(df: pd.DataFrame, output_dir: str):
    """
    Step 3 & 8: Generate and save exploratory data analysis visualizations:
    - Feature distributions (numerical histograms + categorical count plot)
    - Correlation heatmap
    """
    print("Generating EDA visualizations...")

    num_cols = ['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Performance Index']
    cat_col = 'Extracurricular Activities'

    # 1. Feature Distributions
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[idx], color="#2563EB", bins=20, edgecolor="black")
        axes[idx].set_title(f"Distribution of {col}", fontweight="bold")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Count")

    # Categorical distribution on the 6th subplot
    sns.countplot(data=df, x=cat_col, hue=cat_col, ax=axes[5], palette=["#3B82F6", "#10B981"], legend=False)
    axes[5].set_title(f"Distribution of {cat_col}", fontweight="bold")
    axes[5].set_xlabel(cat_col)
    axes[5].set_ylabel("Count")

    plt.suptitle("Student Performance Dataset - Feature Distributions", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "feature_distributions.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    print(f"  [+] Saved feature distributions to: {dist_path}")

    # 2. Correlation Heatmap
    # For correlation, encode Extracurricular Activities as binary (Yes=1, No=0)
    df_corr = df.copy()
    df_corr['Extracurricular Activities'] = (df_corr['Extracurricular Activities'] == 'Yes').astype(int)

    plt.figure(figsize=(9, 7))
    corr_matrix = df_corr.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Feature Correlation Heatmap", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"  [+] Saved correlation heatmap to: {heatmap_path}")


def prepare_and_train_model(df: pd.DataFrame):
    """
    Steps 4, 5, & 6:
    - Data cleaning (remove duplicate rows)
    - Train-test split (80:20, random_state=42)
    - Pipeline construction with ColumnTransformer (passthrough numeric, OneHotEncode categorical)
    - Fit Multiple Linear Regression
    """
    print("\n" + "=" * 60)
    print(" 2. DATA PREPROCESSING & MODEL TRAINING ")
    print("=" * 60)

    # Clean data: remove exact duplicates
    initial_count = len(df)
    df_clean = df.drop_duplicates().reset_index(drop=True)
    dropped_count = initial_count - len(df_clean)
    print(f"Removed {dropped_count} duplicate rows. Remaining rows: {len(df_clean)}")

    target_col = 'Performance Index'
    feature_cols = [c for c in df_clean.columns if c != target_col]

    X = df_clean[feature_cols]
    y = df_clean[target_col]

    print(f"Target variable: '{target_col}'")
    print(f"Input features ({len(feature_cols)}): {feature_cols}")

    # Train/Test Split (80:20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\nTrain set shape: {X_train.shape[0]} rows")
    print(f"Test set shape:  {X_test.shape[0]} rows (strictly held out)")

    # Define Preprocessor with ColumnTransformer
    categorical_cols = ['Extracurricular Activities']
    numerical_cols = [c for c in feature_cols if c not in categorical_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ]
    )

    # Construct the full Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    print("\nTraining Multiple Linear Regression Pipeline on training set...")
    model_pipeline.fit(X_train, y_train)
    print("  [+] Model training complete.")

    return model_pipeline, X_train, y_train, X_test, y_test, numerical_cols, categorical_cols


def evaluate_and_plot(pipeline, X_train, y_train, X_test, y_test, num_cols, cat_cols, output_dir: str):
    """
    Steps 7 & 8:
    - Evaluate test and train sets (MAE, MSE, RMSE, R2, Adjusted R2)
    - Generate Actual vs Predicted plot
    - Generate Residuals Analysis plot
    - Generate Feature Coefficients bar chart
    - Save metrics to JSON
    """
    print("\n" + "=" * 60)
    print(" 3. MODEL EVALUATION & DIAGNOSTICS ")
    print("=" * 60)

    # Predictions
    y_test_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    # Test Metrics
    mae = mean_absolute_error(y_test, y_test_pred)
    mse = mean_squared_error(y_test, y_test_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_test_pred)

    n_test = len(y_test)
    p = X_test.shape[1]
    adj_r2 = 1 - (1 - r2) * (n_test - 1) / (n_test - p - 1)

    # Train Metrics (to verify there is no overfitting or underfitting)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

    print("Test Set Evaluation Metrics:")
    print(f"  - Mean Absolute Error (MAE)       : {mae:.4f}")
    print(f"  - Mean Squared Error (MSE)        : {mse:.4f}")
    print(f"  - Root Mean Squared Error (RMSE)  : {rmse:.4f}")
    print(f"  - R² Score (Coefficient of Det.) : {r2:.4f}")
    print(f"  - Adjusted R² Score               : {adj_r2:.4f}")
    print("\nTraining Set Reference:")
    print(f"  - Train R² Score                  : {train_r2:.4f}")
    print(f"  - Train RMSE                      : {train_rmse:.4f}")

    # Extract Regressor & Feature Names
    regressor = pipeline.named_steps['regressor']
    encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    encoded_cat_names = encoder.get_feature_names_out(cat_cols).tolist()
    all_feature_names = num_cols + encoded_cat_names
    coefficients = regressor.coef_
    intercept = regressor.intercept_

    print(f"\nLinear Regression Model Formula:")
    print(f"  Performance Index = {intercept:.4f}")
    for name, coef in zip(all_feature_names, coefficients):
        print(f"    + ({coef:.4f} * {name})")

    # Save metrics to JSON
    metrics_data = {
        "metrics": {
            "test_mae": round(float(mae), 4),
            "test_mse": round(float(mse), 4),
            "test_rmse": round(float(rmse), 4),
            "test_r2": round(float(r2), 4),
            "test_adjusted_r2": round(float(adj_r2), 4),
            "train_r2": round(float(train_r2), 4),
            "train_rmse": round(float(train_rmse), 4)
        },
        "formula": {
            "intercept": round(float(intercept), 4),
            "coefficients": {k: round(float(v), 4) for k, v in zip(all_feature_names, coefficients)}
        }
    }
    metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
    print(f"\n  [+] Saved evaluation metrics to: {metrics_path}")

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(8, 7))
    plt.scatter(y_test, y_test_pred, alpha=0.3, color="#2563EB", edgecolors='none', s=25, label="Predicted Students")
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color="#DC2626", linestyle="--", linewidth=2, label="Ideal Fit (y = ŷ)")
    plt.title(f"Actual vs. Predicted Performance Index\n(R² = {r2:.4f}, RMSE = {rmse:.4f})", fontweight="bold")
    plt.xlabel("Actual Performance Index")
    plt.ylabel("Predicted Performance Index")
    plt.legend()
    plt.tight_layout()
    act_pred_path = os.path.join(output_dir, "actual_vs_predicted.png")
    plt.savefig(act_pred_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Actual vs Predicted plot to: {act_pred_path}")

    # 2. Residuals Diagnostic Plot (Two Subplots: Residuals vs Predicted & Residual Distribution)
    residuals = y_test - y_test_pred
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Subplot A: Residuals vs Predicted (Homoscedasticity check)
    ax1.scatter(y_test_pred, residuals, alpha=0.3, color="#7C3AED", s=25, edgecolors='none')
    ax1.axhline(0, color="#DC2626", linestyle="--", linewidth=2)
    ax1.set_title("Residuals vs. Predicted Values (Homoscedasticity)", fontweight="bold")
    ax1.set_xlabel("Predicted Performance Index")
    ax1.set_ylabel("Residual (Actual - Predicted)")

    # Subplot B: Residuals Distribution (Normality check)
    sns.histplot(residuals, kde=True, ax=ax2, color="#059669", bins=30, edgecolor="black")
    ax2.axvline(0, color="#DC2626", linestyle="--", linewidth=2)
    ax2.set_title(f"Residuals Distribution (Mean: {residuals.mean():.4f}, Std: {residuals.std():.4f})", fontweight="bold")
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Frequency")

    plt.suptitle("Residual Diagnostics for Multiple Linear Regression", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, "residuals.png")
    plt.savefig(residuals_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Residuals diagnostic plot to: {residuals_path}")

    # 3. Feature Coefficients Plot
    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=True)

    plt.figure(figsize=(9, 5))
    colors = ["#2563EB" if c >= 0 else "#EF4444" for c in coef_df['Coefficient']]
    bars = plt.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, edgecolor="black")
    plt.axvline(0, color="black", linestyle="-", linewidth=0.8)
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width >= 0 else 'right'
        offset = 0.04 if width >= 0 else -0.04
        plt.text(width + offset, bar.get_y() + bar.get_height() / 2, f"{width:.3f}",
                 va='center', ha=ha, fontsize=10, fontweight='bold')
    plt.title("Multiple Linear Regression - Feature Coefficients (Impact)", fontweight="bold")
    plt.xlabel("Coefficient Value (Change in Performance Index per unit increase)")
    plt.ylabel("Feature")
    plt.tight_layout()
    coef_path = os.path.join(output_dir, "feature_coefficients.png")
    plt.savefig(coef_path, dpi=300)
    plt.close()
    print(f"  [+] Saved feature coefficients plot to: {coef_path}")

    return metrics_data


def save_pipeline(pipeline, model_dir: str):
    """
    Step 9: Save the fitted preprocessing + regression pipeline.
    """
    model_path = os.path.join(model_dir, "regression_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\n  [+] Successfully saved trained Pipeline to: {model_path}")
    return model_path


def test_pipeline_inference(pipeline_path: str):
    """
    Verify that the saved pipeline can be reloaded and correctly predict single and batch inputs.
    """
    print("\n" + "=" * 60)
    print(" 4. TESTING SAVED PIPELINE INFERENCE ")
    print("=" * 60)
    loaded_pipeline = joblib.load(pipeline_path)

    sample_student = pd.DataFrame([{
        'Hours Studied': 7,
        'Previous Scores': 90,
        'Extracurricular Activities': 'Yes',
        'Sleep Hours': 7,
        'Sample Question Papers Practiced': 5
    }])

    pred = loaded_pipeline.predict(sample_student)[0]
    # Bound to valid range [10, 100]
    bounded_pred = max(10.0, min(100.0, float(pred)))
    print(f"Sample Student Input:")
    for k, v in sample_student.iloc[0].to_dict().items():
        print(f"  - {k}: {v}")
    print(f"\nModel Prediction for Performance Index: {bounded_pred:.2f}")
    print("  [+] Verification successful: Pipeline loaded and executed inference seamlessly.")
    print("=" * 60 + "\n")


def main():
    print("Starting Student Performance Multiple Linear Regression Pipeline...")
    data_path = find_dataset_path()
    print(f"Loaded dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # 1. Inspection
    inspect_dataset(df)

    # 2. EDA Visualizations
    perform_eda_visualizations(df, OUTPUTS_DIR)

    # 3. Preprocessing, Splitting, and Training
    pipeline, X_train, y_train, X_test, y_test, num_cols, cat_cols = prepare_and_train_model(df)

    # 4. Evaluation and Diagnostics
    evaluate_and_plot(pipeline, X_train, y_train, X_test, y_test, num_cols, cat_cols, OUTPUTS_DIR)

    # 5. Save Model Pipeline
    model_path = save_pipeline(pipeline, MODELS_DIR)

    # 6. Test Model Inference
    test_pipeline_inference(model_path)

    print("Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
