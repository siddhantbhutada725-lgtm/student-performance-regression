"""
Student Performance Prediction - Interactive Streamlit Web Application
======================================================================
This application serves predictions for the Multiple Linear Regression
model trained to forecast a student's academic 'Performance Index'.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# Suppress harmless BLAS/Accelerate matmul warnings on macOS ARM
warnings.filterwarnings('ignore', message='.*encountered in matmul.*')

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor | Multiple Linear Regression",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for a Professional UI
st.markdown("""
<style>
    /* Metric Cards */
    .metric-box {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.95rem;
        opacity: 0.85;
        margin-top: 6px;
    }
    .formula-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 14px 18px;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.95rem;
        color: #1E293B;
        margin: 15px 0;
    }
    .info-card {
        background-color: #F1F5F9;
        padding: 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        border: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "regression_model.joblib")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "evaluation_metrics.json")


@st.cache_resource
def load_trained_model():
    """Load the pre-trained Scikit-Learn Pipeline from disk."""
    if not os.path.isfile(MODEL_PATH):
        return None
    try:
        pipeline = joblib.load(MODEL_PATH)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model from {MODEL_PATH}: {str(e)}")
        return None


def load_metrics():
    """Load saved evaluation metrics and regression formula."""
    if os.path.isfile(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def get_performance_tier(score: float):
    """Categorize the student score into qualitative performance bands."""
    if score >= 85:
        return "🌟 Outstanding Performance", "#10B981", "The student is achieving top-tier academic results."
    elif score >= 70:
        return "📈 Good / Strong Performance", "#3B82F6", "The student is performing solidly above average."
    elif score >= 50:
        return "⚖️ Moderate / Average Performance", "#F59E0B", "The student has acceptable performance with room for growth."
    else:
        return "⚠️ Needs Academic Support", "#EF4444", "Focused tutoring, more study hours, and practice are recommended."


def main():
    pipeline = load_trained_model()
    metrics_data = load_metrics()

    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/clouds/200/graduation-cap.png", width=90)
        st.title("Student Analytics")
        st.markdown("**Multiple Linear Regression** on Kaggle Student Performance Dataset.")

        st.markdown("---")
        st.subheader("🎯 Student Preset Profiles")
        st.caption("Select a preset to populate test values quickly:")

        preset = st.radio(
            "Choose a Profile:",
            ("Custom", "Top Achiever", "Consistent Student", "Under-Prepared Student"),
            index=0
        )

        preset_values = {
            "Custom": (5, 70, "Yes", 7, 5),
            "Top Achiever": (8, 95, "Yes", 8, 8),
            "Consistent Student": (6, 75, "Yes", 7, 4),
            "Under-Prepared Student": (2, 45, "No", 5, 1)
        }

        default_hours, default_prev, default_extra, default_sleep, default_papers = preset_values[preset]

        st.markdown("---")
        st.markdown("### 🏆 Model Quick Stats")
        if metrics_data:
            m = metrics_data.get("metrics", {})
            st.metric("Test R² Score", f"{m.get('test_r2', 0):.4f}")
            st.metric("Test MAE", f"{m.get('test_mae', 0):.2f} pts")
            st.metric("Test RMSE", f"{m.get('test_rmse', 0):.2f} pts")
        else:
            st.info("Train the model using `python3 main.py` to view metrics.")

    # Header Section
    st.title("🎓 Student Academic Performance Predictor")
    st.markdown(
        "Estimate a student's **Performance Index (10 – 100)** based on their daily study habits, "
        "historical scores, sleep patterns, and practice paper preparation using **Multiple Linear Regression**."
    )

    if pipeline is None:
        st.warning("⚠️ **Trained model pipeline not found!**")
        st.markdown(
            "Please run the training script in your terminal to train and save the pipeline:\n"
            "```bash\npython3 main.py\n```"
        )
        if st.button("🚀 Train Model Now"):
            with st.spinner("Training model pipeline..."):
                import subprocess
                res = subprocess.run(["python3", os.path.join(BASE_DIR, "main.py")], capture_output=True, text=True)
                if res.returncode == 0:
                    st.success("Model trained successfully! Reloading...")
                    st.rerun()
                else:
                    st.error(f"Training failed: {res.stderr}")
        return

    # Tabs
    tab_predict, tab_analytics, tab_diagnostics, tab_about = st.tabs([
        "🔮 Performance Prediction",
        "📊 Model Evaluation & Metrics",
        "📈 Visual Diagnostic Gallery",
        "📖 Algorithm & Dataset Details"
    ])

    # TAB 1: PREDICTION
    with tab_predict:
        st.subheader("Enter Student Academic & Lifestyle Attributes")
        st.caption("All input fields include real-world boundaries matching the dataset distribution.")

        with st.form("prediction_form"):
            col1, col2 = st.columns(2)

            with col1:
                hours_studied = st.slider(
                    "⏱️ Hours Studied Daily",
                    min_value=1,
                    max_value=9,
                    value=int(default_hours),
                    step=1,
                    help="Total number of dedicated hours the student studies per day (1 to 9)."
                )

                previous_scores = st.slider(
                    "📝 Previous Exam Scores",
                    min_value=40,
                    max_value=99,
                    value=int(default_prev),
                    step=1,
                    help="Average percentage or score earned in previous tests (40 to 99)."
                )

                extra_activities = st.selectbox(
                    "⚽ Extracurricular Activities",
                    options=["Yes", "No"],
                    index=0 if default_extra == "Yes" else 1,
                    help="Participation in athletics, arts, clubs, or academic competitions."
                )

            with col2:
                sleep_hours = st.slider(
                    "🌙 Daily Sleep Hours",
                    min_value=4,
                    max_value=9,
                    value=int(default_sleep),
                    step=1,
                    help="Average daily hours of sleep obtained by the student (4 to 9)."
                )

                sample_papers = st.slider(
                    "📄 Mock / Sample Papers Practiced",
                    min_value=0,
                    max_value=9,
                    value=int(default_papers),
                    step=1,
                    help="Total number of sample question papers practiced (0 to 9)."
                )

                st.markdown("<br>", unsafe_allow_html=True)
                predict_button = st.form_submit_button(
                    "🚀 Compute Predicted Performance Index",
                    type="primary",
                    use_container_width=True
                )

        if predict_button:
            # Prepare Input DataFrame matching exact training feature names & types
            input_df = pd.DataFrame([{
                "Hours Studied": int(hours_studied),
                "Previous Scores": int(previous_scores),
                "Extracurricular Activities": str(extra_activities),
                "Sleep Hours": int(sleep_hours),
                "Sample Question Papers Practiced": int(sample_papers)
            }])

            try:
                raw_pred = pipeline.predict(input_df)[0]
                # Bound predictions between [10.0, 100.0] as defined by the dataset scale
                bounded_pred = max(10.0, min(100.0, float(raw_pred)))
                tier_label, tier_color, tier_desc = get_performance_tier(bounded_pred)

                st.markdown("---")
                st.subheader("🎯 Prediction Results")

                res_col1, res_col2 = st.columns([1, 1.3])

                with res_col1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <div class="metric-title">Predicted Performance Index</div>
                        <div class="metric-value">{bounded_pred:.1f} <span style="font-size: 1.5rem; opacity: 0.8;">/ 100</span></div>
                        <div class="metric-sub">{tier_label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.progress(bounded_pred / 100.0)
                    st.caption(f"Status: {tier_desc}")

                with res_col2:
                    st.markdown("#### 🔍 Feature Contribution Breakdown")
                    st.markdown(
                        "Based on the learned linear regression coefficients, here is how each attribute influences this prediction:"
                    )

                    # Dynamic breakdown
                    reg = pipeline.named_steps['regressor']
                    enc = pipeline.named_steps['preprocessor'].named_transformers_['cat']
                    cat_name = enc.get_feature_names_out(['Extracurricular Activities'])[0]

                    coef_hours = reg.coef_[0]
                    coef_prev = reg.coef_[1]
                    coef_sleep = reg.coef_[2]
                    coef_papers = reg.coef_[3]
                    coef_extra = reg.coef_[4]
                    intercept = reg.intercept_

                    extra_val = 1.0 if extra_activities == "Yes" else 0.0

                    contrib_hours = coef_hours * hours_studied
                    contrib_prev = coef_prev * previous_scores
                    contrib_sleep = coef_sleep * sleep_hours
                    contrib_papers = coef_papers * sample_papers
                    contrib_extra = coef_extra * extra_val

                    contrib_df = pd.DataFrame([
                        {"Factor": "Baseline Intercept", "Value": f"{intercept:.2f}", "Points Impact": f"{intercept:+.2f}"},
                        {"Factor": "Previous Scores", "Value": f"{previous_scores} pts", "Points Impact": f"{contrib_prev:+.2f}"},
                        {"Factor": "Hours Studied", "Value": f"{hours_studied} hrs", "Points Impact": f"{contrib_hours:+.2f}"},
                        {"Factor": "Sleep Hours", "Value": f"{sleep_hours} hrs", "Points Impact": f"{contrib_sleep:+.2f}"},
                        {"Factor": "Sample Papers", "Value": f"{sample_papers} papers", "Points Impact": f"{contrib_papers:+.2f}"},
                        {"Factor": "Extracurriculars", "Value": extra_activities, "Points Impact": f"{contrib_extra:+.2f}"},
                    ])

                    st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                    st.info(f"💡 **Key Insight:** Previous Scores (+{contrib_prev:.1f} pts) and Study Hours (+{contrib_hours:.1f} pts) are the strongest academic drivers.")

            except Exception as e:
                st.error(f"Prediction failed with error: {str(e)}")

    # TAB 2: MODEL EVALUATION & METRICS
    with tab_analytics:
        st.subheader("📊 Model Evaluation & Statistical Performance")
        st.markdown(
            "The Multiple Linear Regression model was evaluated on a held-out test set "
            "(20% split, 1,975 student records) with `random_state=42` to ensure objective, un-leaked scoring."
        )

        if metrics_data:
            m = metrics_data.get("metrics", {})
            f = metrics_data.get("formula", {})

            col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
            col_m1.metric("R² Score", f"{m.get('test_r2', 0):.4f}", help="Proportion of variance explained by features.")
            col_m2.metric("Adjusted R²", f"{m.get('test_adjusted_r2', 0):.4f}", help="Penalizes non-informative features.")
            col_m3.metric("MAE", f"{m.get('test_mae', 0):.4f}", help="Average absolute error in index points.")
            col_m4.metric("MSE", f"{m.get('test_mse', 0):.4f}", help="Mean squared error.")
            col_m5.metric("RMSE", f"{m.get('test_rmse', 0):.4f}", help="Root mean squared error in index units.")

            st.markdown("---")
            st.subheader("📐 Learned Mathematical Regression Formula")
            intercept = f.get("intercept", 0)
            coefs = f.get("coefficients", {})

            st.markdown(f"""
            <div class="formula-box">
                <b>Performance Index</b> = {intercept:.4f}<br>
                &nbsp;&nbsp;+ ({coefs.get('Hours Studied', 0):.4f} × Hours Studied)<br>
                &nbsp;&nbsp;+ ({coefs.get('Previous Scores', 0):.4f} × Previous Scores)<br>
                &nbsp;&nbsp;+ ({coefs.get('Sleep Hours', 0):.4f} × Sleep Hours)<br>
                &nbsp;&nbsp;+ ({coefs.get('Sample Question Papers Practiced', 0):.4f} × Sample Question Papers Practiced)<br>
                &nbsp;&nbsp;+ ({coefs.get('Extracurricular Activities_Yes', 0):.4f} × Extracurricular Activities [Yes=1, No=0])
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### 💡 Plain-English Interpretation of Coefficients")
            st.markdown(f"""
            - **Previous Scores ({coefs.get('Previous Scores', 0):.4f}):** For every 1-point increase in past exams, the Performance Index rises by ~**{coefs.get('Previous Scores', 0):.2f}** points.
            - **Hours Studied ({coefs.get('Hours Studied', 0):.4f}):** Every additional hour studied per day yields a substantial **+{coefs.get('Hours Studied', 0):.2f}** point increase.
            - **Extracurricular Activities ({coefs.get('Extracurricular Activities_Yes', 0):.4f}):** Active participation adds ~**+{coefs.get('Extracurricular Activities_Yes', 0):.2f}** points.
            - **Sleep Hours ({coefs.get('Sleep Hours', 0):.4f}):** Each healthy hour of rest contributes ~**+{coefs.get('Sleep Hours', 0):.2f}** points.
            - **Sample Question Papers ({coefs.get('Sample Question Papers Practiced', 0):.4f}):** Each mock paper practiced contributes ~**+{coefs.get('Sample Question Papers Practiced', 0):.2f}** points.
            """)

    # TAB 3: VISUAL DIAGNOSTICS GALLERY
    with tab_diagnostics:
        st.subheader("📈 Visual Diagnostic & Exploratory Plots")
        st.caption("Visual validation of linear regression assumptions (linearity, homoscedasticity, residual normality).")

        img_actual = os.path.join(OUTPUTS_DIR, "actual_vs_predicted.png")
        img_residuals = os.path.join(OUTPUTS_DIR, "residuals.png")
        img_heatmap = os.path.join(OUTPUTS_DIR, "correlation_heatmap.png")
        img_dist = os.path.join(OUTPUTS_DIR, "feature_distributions.png")
        img_coef = os.path.join(OUTPUTS_DIR, "feature_coefficients.png")

        diag_col1, diag_col2 = st.columns(2)

        with diag_col1:
            st.markdown("#### 1. Actual vs. Predicted Values")
            if os.path.isfile(img_actual):
                st.image(img_actual, use_container_width=True, caption="Scatter plot of true vs predicted values with ideal y = x reference line.")
            else:
                st.info("Run `python3 main.py` to generate this plot.")

            st.markdown("#### 2. Feature Correlation Heatmap")
            if os.path.isfile(img_heatmap):
                st.image(img_heatmap, use_container_width=True, caption="Correlation matrix showing high correlation with Previous Scores (0.915) and Study Hours (0.374).")
            else:
                st.info("Run `python3 main.py` to generate this plot.")

        with diag_col2:
            st.markdown("#### 3. Residual Diagnostics (Homoscedasticity & Normality)")
            if os.path.isfile(img_residuals):
                st.image(img_residuals, use_container_width=True, caption="Residual scatter (constant variance) and distribution (bell-shaped centered at 0).")
            else:
                st.info("Run `python3 main.py` to generate this plot.")

            st.markdown("#### 4. Feature Coefficients & Impact")
            if os.path.isfile(img_coef):
                st.image(img_coef, use_container_width=True, caption="Marginal effect of each feature on the Performance Index.")
            else:
                st.info("Run `python3 main.py` to generate this plot.")

        st.markdown("---")
        st.markdown("#### 5. Raw Feature Distributions")
        if os.path.isfile(img_dist):
            st.image(img_dist, use_container_width=True, caption="Histograms and count plots of all features across 10,000 student records.")

    # TAB 4: ABOUT
    with tab_about:
        st.subheader("📖 About Multiple Linear Regression & The Dataset")
        st.markdown("""
        ### What is Multiple Linear Regression?
        **Multiple Linear Regression (MLR)** models the linear relationship between a continuous dependent target variable ($Y$) and two or more independent explanatory features ($X_1, X_2, \\dots, X_p$):

        $$Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\dots + \\beta_p X_p + \\epsilon$$

        - **$\\beta_0$ (Intercept):** The baseline value of $Y$ when all input features are zero.
        - **$\\beta_i$ (Slope / Coefficient):** The change in $Y$ for every one-unit increase in $X_i$, keeping all other features constant (*ceteris paribus*).
        - **$\\epsilon$ (Error term):** Unobserved random noise.

        ---

        ### Key Assumptions of Linear Regression
        1. **Linearity:** The relationship between predictors and the target is linear.
        2. **Independence:** Observations are independent of one another.
        3. **Homoscedasticity:** The variance of residuals remains constant across all predicted values.
        4. **Normality:** The residuals are approximately normally distributed with zero mean.
        5. **No Multicollinearity:** Independent variables are not strongly correlated with each other.

        ---

        ### Preprocessing Pipeline Architecture
        To prevent **data leakage** and ensure seamless inference:
        - Numerical features are handled using `passthrough`.
        - Categorical feature `Extracurricular Activities` is transformed with `OneHotEncoder(drop='first')`.
        - The preprocessing and regression stages are unified inside a single **Scikit-Learn Pipeline**, exported to disk via `joblib`.
        """)


if __name__ == "__main__":
    main()
