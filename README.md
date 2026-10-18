# 🎓 Student Performance Prediction using Multiple Linear Regression

An end-to-end, beginner-friendly Machine Learning project in Python that predicts a student's academic **Performance Index** using **Multiple Linear Regression**. 

This repository demonstrates rigorous data science best practices: Exploratory Data Analysis (EDA), zero data leakage preprocessing with Scikit-Learn `Pipeline` and `ColumnTransformer`, statistical model evaluation, residual diagnostic plots, and an interactive **Streamlit** web application.

---

## 📌 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description & Inspection](#-dataset-description--inspection)
3. [Architecture & Workflow](#-architecture--workflow)
4. [Machine Learning Theory & Assumptions](#-machine-learning-theory--assumptions)
5. [Model Training & Evaluation Results](#-model-training--evaluation-results)
6. [Interactive Web Application](#-interactive-web-application)
7. [Project File Structure](#-project-file-structure)
8. [Installation & Setup](#-installation--setup)
9. [How to Run](#-how-to-run)
10. [Key Takeaways for Beginners](#-key-takeaways-for-beginners)

---

## 🔍 Project Overview

The objective of this project is to model and predict student performance based on their daily study habits, previous exam scores, sleep duration, practice mock tests, and extracurricular activities.

- **Algorithm:** Multiple Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Key Metric:** $R^2 \approx 0.9884$ (explains ~98.8% of test score variance)
- **Deployment:** Interactive Streamlit web interface with dynamic coefficient breakdown and diagnostic visuals.

---

## 📊 Dataset Description & Inspection

The dataset is Kaggle's **Student Performance (Multiple Linear Regression)** dataset containing **10,000 student observations**:

| Feature Name | Type | Description | Range / Values |
| :--- | :--- | :--- | :--- |
| `Hours Studied` | Numerical (`int64`) | Total hours spent studying daily | 1 to 9 hours |
| `Previous Scores` | Numerical (`int64`) | Previous exam percentage/marks | 40 to 99 marks |
| `Extracurricular Activities` | Categorical (`object`) | Student participation in activities | `Yes` or `No` |
| `Sleep Hours` | Numerical (`int64`) | Average hours of daily sleep | 4 to 9 hours |
| `Sample Question Papers Practiced` | Numerical (`int64`) | Number of mock tests completed | 0 to 9 papers |
| **`Performance Index`** | **Numerical (`float64`)** | **Target variable: Overall performance index** | **10.0 to 100.0** |

### Key Findings from Data Inspection
- **Missing Values:** `0` missing values across all columns.
- **Duplicates:** `127` exact duplicate rows were detected and removed prior to model training to maintain data integrity.
- **Clean Sample Size:** `9,873` unique student records.

---

## 🛠️ Architecture & Workflow

Every major stage in the pipeline serves a specific educational and engineering purpose:

```
┌─────────────────────────────────┐
│     Raw Student Dataset         │
│   (Student_Performance.csv)     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  1. Inspection & Cleaning       │
│  - Verify dtypes & nulls        │
│  - Drop 127 duplicate rows      │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  2. Exploratory Data Analysis   │
│  - Feature distributions        │
│  - Correlation heatmap          │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  3. Train/Test Split (80:20)    │
│  - X_train: 7,898 rows          │
│  - X_test:  1,975 rows (locked) │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  4. ColumnTransformer Pipeline  │
│  - Num: passthrough             │
│  - Cat: OneHotEncoder           │
│  - Reg: LinearRegression        │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  5. Model Evaluation & Plots    │
│  - MAE, MSE, RMSE, R², Adj-R²   │
│  - Actual vs Predicted plot     │
│  - Residuals & Coefficients     │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  6. Model Serialization & App   │
│  - models/regression_model.joblib
│  - Interactive Streamlit app.py │
└─────────────────────────────────┘
```

1. **Inspection & Data Cleaning:** 
   Validates data types and removes duplicate records.
2. **Exploratory Data Analysis (EDA):**
   Reveals correlations between input features and academic achievement (e.g., `Previous Scores` has a strong $0.915$ correlation with the target).
3. **Train-Test Splitting (80:20, `random_state=42`):**
   Splits data *before* fitting encoders to prevent **data leakage**.
4. **Scikit-Learn Pipeline & ColumnTransformer:**
   Encapsulates categorical encoding (`OneHotEncoder(drop='first')`) and regression in a single pipeline. This ensures that new, raw input data can be fed directly to the model during production without manual preprocessing.
5. **Model Evaluation:**
   Calculates MAE, MSE, RMSE, and $R^2$ / Adjusted $R^2$ on the strictly held-out test set.
6. **Diagnostic Plotting:**
   Visualizes residuals and linear regression assumptions.
7. **Model Serialization:**
   Saves the entire fitted pipeline as `models/regression_model.joblib`.
8. **Web Interface:**
   A Streamlit app (`app.py`) for live student performance forecasting.

---

## 📐 Machine Learning Theory & Assumptions

### Multiple Linear Regression Formula
Multiple Linear Regression models the target variable $Y$ as a linear combination of $p$ input features:

$$Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \epsilon$$

Where:
- $\beta_0$: Intercept (value of $Y$ when all features equal $0$).
- $\beta_i$: Regression coefficient (change in $Y$ per unit increase in $X_i$, holding all other features constant).
- $\epsilon$: Error/residual term ($\epsilon \sim \mathcal{N}(0, \sigma^2)$).

### Key Linear Regression Assumptions Checked
1. **Linearity:** Linear relationship between input features and the target.
2. **Independence:** Samples are independent student observations.
3. **Homoscedasticity:** Constant variance of residuals across all predicted values (verified via `outputs/residuals.png`).
4. **Normality of Residuals:** Residual errors follow an approximate Gaussian bell curve centered at zero.
5. **Absence of Multicollinearity:** Low inter-feature correlation among independent predictors.

---

## 📈 Model Training & Evaluation Results

### Real Evaluation Metrics (Test Set, $n = 1,975$)
*Evaluated on the held-out test split with `random_state=42`:*

| Evaluation Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **$R^2$ Score** | **$0.9884$** | Explains **$98.84\%$** of score variance in test students |
| **Adjusted $R^2$** | **$0.9884$** | Penalizes irrelevant features (confirms no penalty) |
| **Mean Absolute Error (MAE)** | **$1.6470$ points** | Average prediction error is only ~1.65 index points |
| **Mean Squared Error (MSE)** | **$4.3059$** | Average squared residual difference |
| **Root Mean Squared Error (RMSE)** | **$2.0751$ points** | Typical standard deviation of unexplained variance |
| **Train $R^2$ Reference** | **$0.9887$** | No overfitting; test and train performance are virtually identical |

### Learned Model Equation
$$\text{Performance Index} = -33.9813 + 2.8510(\text{Hours}) + 1.0184(\text{Previous Scores}) + 0.4721(\text{Sleep}) + 0.1887(\text{Papers}) + 0.5738(\text{Extracurriculars})$$

### Interpretation of Feature Coefficients
- **Hours Studied ($+2.851$):** Each additional hour studied per day increases the predicted score by **$+2.85$ points**.
- **Previous Scores ($+1.018$):** Every 1-point increase in past exams translates into **$+1.02$ points**.
- **Extracurricular Activities ($+0.574$):** Participating in activities provides a **$+0.57$ point** boost.
- **Sleep Hours ($+0.472$):** Each hour of healthy sleep contributes **$+0.47$ points**.
- **Sample Question Papers ($+0.189$):** Each practice paper completed adds **$+0.19$ points**.

### Generated Diagnostic Visualizations (saved in `outputs/`)
- `actual_vs_predicted.png`: Demonstrates tightly clustered points around the ideal 45-degree line ($y = \hat{y}$).
- `residuals.png`: Shows homoscedastic residual scatter and a symmetric, normal distribution centered at $0$.
- `correlation_heatmap.png`: Correlation matrix of all features.
- `feature_distributions.png`: Univariate distributions of each feature.
- `feature_coefficients.png`: Ranked horizontal bar chart of feature coefficients.

---

## 💻 Interactive Web Application

The interactive web interface is built with **Streamlit** (`app.py`). It includes:
1. **Interactive Sliders & Inputs:** Bounded within real dataset distributions.
2. **Student Preset Profiles:** Instant loading of profiles (*Top Achiever*, *Consistent Student*, *Under-Prepared Student*).
3. **Live Prediction & Performance Tiers:** Categorizes predictions into *Outstanding*, *Strong*, *Average*, and *Needs Support*.
4. **Dynamic Feature Breakdown:** Explains exactly how many points each feature added or subtracted to produce the final score.
5. **Built-in Analytics & Diagnostics:** Displays evaluation metrics, model formulas, and diagnostic plots directly in the web UI.

---

## 📁 Project File Structure

```text
student-performace-regression/
├── Student_Performance.csv           # Kaggle student performance dataset
├── Student_Performace.csv           # Mirror dataset alias
├── main.py                          # Full training, EDA, evaluation & export script
├── app.py                           # Streamlit interactive web application
├── requirements.txt                 # Project Python dependencies
├── requirement.txt                  # Compatibility alias
├── .gitignore                       # Git ignore configuration
├── models/
│   └── regression_model.joblib      # Fitted preprocessing + regression pipeline
└── outputs/
    ├── actual_vs_predicted.png      # Actual vs Predicted scatter plot
    ├── correlation_heatmap.png      # Feature correlation heatmap
    ├── evaluation_metrics.json      # Quantitative evaluation metrics
    ├── feature_coefficients.png     # Linear regression coefficients bar chart
    ├── feature_distributions.png    # Univariate feature histograms
    └── residuals.png                # Residuals diagnostic (scatter & normality)
```

---

## ⚡ Installation & Setup

### 1. Clone or Open the Repository
```bash
cd /Users/sidhant/student-performace-regression
```

### 2. (Optional) Create and Activate a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

### Step 1: Run the Training & Evaluation Pipeline
Execute `main.py` to inspect the CSV, generate EDA plots, train the Multiple Linear Regression pipeline, evaluate performance, and save the model:

```bash
python3 main.py
```

**Expected Output:**
```text
============================================================
 1. DATASET INSPECTION & SUMMARY 
============================================================
Shape: 10000 rows, 6 columns
...
Removed 127 duplicate rows. Remaining rows: 9873
Train set shape: 7898 rows
Test set shape:  1975 rows (strictly held out)

Test Set Evaluation Metrics:
  - Mean Absolute Error (MAE)       : 1.6470
  - Mean Squared Error (MSE)        : 4.3059
  - Root Mean Squared Error (RMSE)  : 2.0751
  - R² Score (Coefficient of Det.) : 0.9884
  - Adjusted R² Score               : 0.9884

Linear Regression Model Formula:
  Performance Index = -33.9813
    + (2.8510 * Hours Studied)
    + (1.0184 * Previous Scores)
    + (0.4721 * Sleep Hours)
    + (0.1887 * Sample Question Papers Practiced)
    + (0.5738 * Extracurricular Activities_Yes)

  [+] Successfully saved trained Pipeline to: models/regression_model.joblib
```

### Step 2: Launch the Streamlit Web Application
Run the Streamlit application:

```bash
streamlit run app.py
```

Once launched, Streamlit will provide a local URL:
```text
Local URL: http://localhost:8501
Network URL: http://...
```
Open `http://localhost:8501` in your browser to interact with the student performance predictor.

---

## 🧠 Key Takeaways for Beginners

1. **Why use a Pipeline?**  
   Combining the `ColumnTransformer` (for One-Hot Encoding) with `LinearRegression` into a single pipeline prevents data leakage during cross-validation/testing and ensures your production app can take raw user inputs without needing separate preprocessing code.
2. **Why evaluate both Train and Test R²?**  
   If your model scores $0.99$ on train and $0.60$ on test, it is *overfitting*. Because both our train ($0.9887$) and test ($0.9884$) scores are virtually identical, we know the model generalizes reliably.
3. **Why check Residuals?**  
   Residual plots confirm whether the linear regression assumptions are met. The random, uniform distribution centered around $0$ in `outputs/residuals.png` validates that linear regression is an appropriate choice for this data.
