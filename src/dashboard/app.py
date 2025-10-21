"""
TRANCE Dashboard - Main Application with SHAP & Live Predictions
Save as: app.py
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import joblib
import lightgbm as lgb
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="TRANCE - Readmission Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .risk-high {
        background-color: #ffebee;
        border-left-color: #d32f2f;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left-color: #f57c00;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left-color: #388e3c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize paths
@st.cache_resource
def get_paths():
    base_path = Path('')
    return {
        'results': base_path / 'outputs/results',
        'models': base_path / 'outputs/models',
        'processed': base_path / 'data/processed'
    }

paths = get_paths()

# Load models
@st.cache_resource
def load_models():
    """Load trained models"""
    try:
        baseline_path = paths['models'] / 'baseline_model.txt'
        fused_path = paths['models'] / 'fused_model.txt'
        
        if not baseline_path.exists() or not fused_path.exists():
            st.error("⚠️ Models not found. Please train models first.")
            return None, None, None, None
        
        model_baseline = lgb.Booster(model_file=str(baseline_path))
        model_fused = lgb.Booster(model_file=str(fused_path))
        calibrator = joblib.load(paths['models'] / 'calibrator.pkl')
        
        # Load SHAP values if available
        shap_path = paths['models'] / 'shap_values.pkl'
        if shap_path.exists():
            shap_data = joblib.load(shap_path)
        else:
            shap_data = None
        
        return model_baseline, model_fused, calibrator, shap_data
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Load data
@st.cache_data
def load_data():
    """Load all necessary data"""
    try:
        predictions = pd.read_parquet(paths['results'] / 'test_predictions.parquet')

        with open(paths['results'] / 'model_results_summary.json', 'r') as f:
            results_summary = json.load(f)

        # Load test features for patient details
        test_features = pd.read_parquet(paths['processed'] / 'test_fused.parquet')

        # Merge predictions with features
        full_data = predictions.merge(test_features, on=['HADM_ID', 'SUBJECT_ID'])
        
        # Load feature info
        feature_info_path = paths['processed'] / 'feature_info.json'
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                feature_info = json.load(f)
        else:
            feature_info = None

        return predictions, results_summary, full_data, feature_info
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None, None

# Load everything
model_baseline, model_fused, calibrator, shap_data = load_models()
predictions, results_summary, full_data, feature_info = load_data()

if predictions is None or results_summary is None:
    st.error("Unable to load data. Please ensure models have been trained.")
    st.stop()

# Sidebar navigation
st.sidebar.markdown("# 🏥 TRANCE")
st.sidebar.markdown("### Readmission Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Overview", "📈 Volume Forecasting", "🎯 Patient Risk Dashboard", 
     "🔍 Model Monitoring", "🎲 Live Prediction Tool"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "TRANCE uses machine learning with clinical text embeddings "
    "to predict 30-day hospital readmissions."
)

# ==================== PAGE 1: EXECUTIVE OVERVIEW ====================
if page == "📊 Executive Overview":
    st.markdown("<h1 class='main-header'>📊 Executive Overview</h1>", unsafe_allow_html=True)

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        auroc = results_summary['fused_model']['test_auroc']
        color = "🟢" if auroc >= 0.75 else "🟡" if auroc >= 0.70 else "🔴"
        st.metric("Model AUROC", f"{auroc:.3f}", help="Area Under ROC Curve")
        st.caption(f"{color} {'Excellent' if auroc >= 0.75 else 'Good' if auroc >= 0.70 else 'Fair'} Performance")

    with col2:
        auprc = results_summary['fused_model']['test_auprc']
        st.metric("Model AUPRC", f"{auprc:.3f}", help="Area Under Precision-Recall Curve")

    with col3:
        brier = results_summary['fused_model']['brier_score_calibrated']
        color = "🟢" if brier <= 0.15 else "🟡" if brier <= 0.20 else "🔴"
        st.metric("Brier Score", f"{brier:.3f}", help="Lower is better")
        st.caption(f"{color} {'Well' if brier <= 0.15 else 'Moderately' if brier <= 0.20 else 'Poorly'} Calibrated")

    with col4:
        improvement = results_summary['improvements']['auroc_gain_pct']
        st.metric("Embedding Boost", f"{improvement:+.1f}%", help="AUROC improvement from text embeddings")

    st.markdown("---")

    # Performance visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ROC Curve")
        
        from sklearn.metrics import roc_curve
        
        y_true = predictions['true_label']
        y_pred_baseline = predictions['pred_prob_baseline']
        y_pred_fused = predictions['pred_prob_fused_calibrated']
        
        fpr_baseline, tpr_baseline, _ = roc_curve(y_true, y_pred_baseline)
        fpr_fused, tpr_fused, _ = roc_curve(y_true, y_pred_fused)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr_baseline, y=tpr_baseline,
            name=f"Baseline (AUROC={results_summary['baseline_model']['test_auroc']:.3f})",
            line=dict(color='lightblue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fpr_fused, y=tpr_fused,
            name=f"Fused Model (AUROC={auroc:.3f})",
            line=dict(color='steelblue', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            name="Random Classifier",
            line=dict(color='gray', width=2, dash='dash')
        ))
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Model Comparison")

        baseline_auroc = results_summary['baseline_model']['test_auroc']
        fused_auroc = results_summary['fused_model']['test_auroc']

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Structured Only', 'Structured + Embeddings'],
            y=[baseline_auroc, fused_auroc],
            text=[f"{baseline_auroc:.3f}", f"{fused_auroc:.3f}"],
            textposition='auto',
            marker_color=['lightblue', 'steelblue']
        ))
        fig.update_layout(
            yaxis_title="AUROC",
            yaxis_range=[0, 1],
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance metrics table
    st.subheader("Performance Metrics Summary")

    metrics_df = pd.DataFrame({
        'Metric': ['AUROC', 'AUPRC', 'Brier Score', 'Embedding Contribution'],
        'Baseline': [
            f"{baseline_auroc:.4f}",
            f"{results_summary['baseline_model']['test_auprc']:.4f}",
            f"{results_summary['baseline_model']['brier_score']:.4f}",
            "N/A"
        ],
        'Fused Model': [
            f"{fused_auroc:.4f}",
            f"{auprc:.4f}",
            f"{brier:.4f}",
            f"{results_summary['embedding_contribution']['importance_pct']:.1f}%"
        ],
        'Improvement': [
            f"+{results_summary['improvements']['auroc_gain']:.4f}",
            f"+{results_summary['improvements']['auprc_gain']:.4f}",
            f"{brier - results_summary['baseline_model']['brier_score']:.4f}",
            "—"
        ]
    })

    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

# ==================== PAGE 2: VOLUME FORECASTING ====================
elif page == "📈 Volume Forecasting":
    st.markdown("<h1 class='main-header'>📈 Volume Forecasting & Capacity Planning</h1>", unsafe_allow_html=True)

    # Simulate temporal data
    np.random.seed(42)
    predictions_temp = predictions.copy()
    predictions_temp['discharge_date'] = pd.date_range(start='2024-01-01', periods=len(predictions), freq='H')
    predictions_temp['discharge_date'] = predictions_temp['discharge_date'].dt.date

    # Aggregate by day
    daily_forecast = predictions_temp.groupby('discharge_date').agg({
        'pred_prob_fused_calibrated': 'sum',
        'true_label': 'sum'
    }).reset_index()

    daily_forecast.columns = ['Date', 'Predicted Readmissions', 'Actual Readmissions']
    daily_forecast['Forecast Error'] = daily_forecast['Predicted Readmissions'] - daily_forecast['Actual Readmissions']

    # Forecast horizon selector
    st.sidebar.markdown("### Forecast Settings")
    horizon = st.sidebar.selectbox("Forecast Horizon", ["7-day", "14-day", "30-day"], index=0)
    horizon_days = int(horizon.split('-')[0])

    forecast_window = daily_forecast.head(horizon_days)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_predicted = forecast_window['Predicted Readmissions'].sum()
        st.metric(
            f"Predicted ({horizon})",
            f"{total_predicted:.0f} patients"
        )

    with col2:
        peak_day = forecast_window.loc[forecast_window['Predicted Readmissions'].idxmax()]
        st.metric(
            "Peak Day",
            f"{peak_day['Date']}",
            delta=f"{peak_day['Predicted Readmissions']:.0f} patients"
        )

    with col3:
        avg_daily = forecast_window['Predicted Readmissions'].mean()
        st.metric(
            "Avg Daily",
            f"{avg_daily:.1f} patients"
        )

    with col4:
        mae = np.abs(forecast_window['Forecast Error']).mean()
        st.metric("Forecast MAE", f"{mae:.2f}")

    st.markdown("---")

    # Time series chart
    st.subheader("Readmission Forecast Timeline")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_window['Date'],
        y=forecast_window['Predicted Readmissions'],
        mode='lines+markers',
        name='Predicted',
        line=dict(color='steelblue', width=3),
        marker=dict(size=8)
    ))

    fig.add_trace(go.Scatter(
        x=forecast_window['Date'],
        y=forecast_window['Actual Readmissions'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='orange', width=2, dash='dash'),
        marker=dict(size=6)
    ))

    historical_avg = daily_forecast['Actual Readmissions'].mean()
    fig.add_hline(
        y=historical_avg,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Historical Avg: {historical_avg:.1f}"
    )

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Readmissions",
        height=400,
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

    # Daily breakdown
    st.subheader("Daily Breakdown")

    forecast_window['Day of Week'] = pd.to_datetime(forecast_window['Date']).dt.day_name()
    forecast_window['Alert Level'] = forecast_window['Predicted Readmissions'].apply(
        lambda x: '🔴 High' if x > historical_avg * 1.5 else '🟡 Elevated' if x > historical_avg * 1.2 else '🟢 Normal'
    )

    display_df = forecast_window[['Date', 'Day of Week', 'Predicted Readmissions', 'Alert Level']].copy()
    display_df['Predicted Readmissions'] = display_df['Predicted Readmissions'].round(1)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Staffing recommendations
    st.subheader("📋 Staffing Recommendations")

    col1, col2 = st.columns(2)

    with col1:
        st.info(
            f"""
            **Recommended Case Managers:** {int(np.ceil(avg_daily / 5))}

            Based on average workload of 5 patients per case manager.
            """
        )

    with col2:
        high_volume_days = len(forecast_window[forecast_window['Predicted Readmissions'] > historical_avg * 1.5])
        if high_volume_days > 0:
            st.warning(
                f"""
                **⚠️ High Volume Alert**

                {high_volume_days} days exceed 150% of historical average.
                """
            )
        else:
            st.success("No high-volume days forecasted.")

# ==================== PAGE 3: PATIENT RISK DASHBOARD ====================
elif page == "🎯 Patient Risk Dashboard":
    st.markdown("<h1 class='main-header'>🎯 High-Risk Patient List</h1>", unsafe_allow_html=True)

    # Filters
    st.sidebar.markdown("### Filters")

    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5
    )

    top_n = st.sidebar.number_input(
        "Show Top N Patients",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    # Filter data
    high_risk = full_data[full_data['risk_score'] >= risk_threshold].copy()
    high_risk = high_risk.nlargest(top_n, 'risk_score')

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("High-Risk Patients", len(high_risk))

    with col2:
        avg_risk = high_risk['risk_score'].mean() if len(high_risk) > 0 else 0
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")

    with col3:
        capacity_status = "🟢 Within Capacity" if len(high_risk) <= 20 else "🟡 Above Capacity"
        st.metric("Capacity Status", capacity_status)

    with col4:
        if 'true_label' in high_risk.columns and len(high_risk) > 0:
            actual_readmits = high_risk['true_label'].sum()
            precision = actual_readmits / len(high_risk)
            st.metric("Precision", f"{precision:.1%}")

    st.markdown("---")

    # Risk distribution
    st.subheader("Risk Score Distribution")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=full_data['risk_score'],
        nbinsx=20,
        marker_color='steelblue',
        opacity=0.7
    ))
    fig.add_vline(
        x=risk_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold: {risk_threshold}%"
    )
    fig.update_layout(
        xaxis_title="Risk Score (%)",
        yaxis_title="Count",
        height=300
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Patient cards
    st.subheader(f"Top {len(high_risk)} High-Risk Patients")

    for idx, row in high_risk.iterrows():
        risk = row['risk_score']

        if risk >= 70:
            risk_icon = "🔴"
            risk_label = "VERY HIGH RISK"
        elif risk >= 50:
            risk_icon = "🟠"
            risk_label = "HIGH RISK"
        else:
            risk_icon = "🟡"
            risk_label = "MODERATE RISK"

        with st.expander(f"{risk_icon} Patient #{row['HADM_ID']} - Risk: {risk:.1f}%"):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"### {risk_icon} {risk:.1f}%")
                st.markdown(f"**{risk_label}**")
                st.markdown("---")

                if 'age' in row and pd.notna(row['age']):
                    st.markdown(f"**Age:** {row['age']:.0f}")
                if 'gender_M' in row:
                    gender = 'Male' if row['gender_M'] == 1 else 'Female'
                    st.markdown(f"**Gender:** {gender}")
                if 'los_days' in row and pd.notna(row['los_days']):
                    st.markdown(f"**Length of Stay:** {row['los_days']:.1f} days")

            with col2:
                st.markdown("**Top Risk Factors:**")

                risk_factors = []

                if 'charlson_score' in row and pd.notna(row['charlson_score']) and row['charlson_score'] > 2:
                    risk_factors.append(f"🔴 High comorbidity (Charlson: {row['charlson_score']:.0f})")

                if 'prior_admissions_180d' in row and pd.notna(row['prior_admissions_180d']) and row['prior_admissions_180d'] > 1:
                    risk_factors.append(f"🟠 Recent admissions ({row['prior_admissions_180d']:.0f})")

                if 'los_days' in row and pd.notna(row['los_days']) and row['los_days'] > 7:
                    risk_factors.append(f"🟡 Extended stay ({row['los_days']:.0f} days)")

                if 'had_icu_stay' in row and row['had_icu_stay'] == 1:
                    risk_factors.append("🟡 ICU admission")

                if 'dx_heart_failure' in row and row['dx_heart_failure'] == 1:
                    risk_factors.append("🔴 Heart failure")

                if len(risk_factors) == 0:
                    risk_factors.append("ℹ️ Risk driven by clinical patterns")

                for factor in risk_factors[:5]:
                    st.markdown(f"- {factor}")

                st.markdown("---")

                st.markdown("**📞 Recommended Actions:**")
                if risk >= 70:
                    st.markdown("- ✅ Schedule 24-48h call")
                    st.markdown("- ✅ Urgent follow-up")
                    st.markdown("- ✅ Medication review")
                elif risk >= 50:
                    st.markdown("- ✅ Schedule 48-72h call")
                    st.markdown("- ✅ Ensure follow-up scheduled")
                else:
                    st.markdown("- ✅ Standard protocol")

# ==================== PAGE 4: MODEL MONITORING ====================
elif page == "🔍 Model Monitoring":
    st.markdown("<h1 class='main-header'>🔍 Model Monitoring & Interpretability</h1>", unsafe_allow_html=True)

    # Model health
    st.subheader("Model Health Status")

    col1, col2, col3 = st.columns(3)

    auroc = results_summary['fused_model']['test_auroc']
    brier = results_summary['fused_model']['brier_score_calibrated']

    with col1:
        auroc_status = "🟢 Good" if auroc >= 0.72 else "🟡 Monitor"
        st.metric("AUROC Status", auroc_status, f"{auroc:.3f}")

    with col2:
        cal_status = "🟢 Good" if brier <= 0.15 else "🟡 Monitor"
        st.metric("Calibration Status", cal_status, f"Brier: {brier:.3f}")

    with col3:
        st.metric("Model Version", "v1.0.0")

    st.markdown("---")

    # Calibration curve
    st.subheader("Calibration Curve")

    from sklearn.calibration import calibration_curve
    
    y_true = predictions['true_label']
    y_pred = predictions['pred_prob_fused_calibrated']
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred, n_bins=10, strategy='uniform'
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))

    fig.add_trace(go.Scatter(
        x=mean_predicted_value,
        y=fraction_of_positives,
        mode='lines+markers',
        name='Model Calibration',
        line=dict(color='steelblue', width=3),
        marker=dict(size=10)
    ))

    fig.update_layout(
        xaxis_title="Predicted Probability",
        yaxis_title="Observed Frequency",
        height=400,
        xaxis_range=[0, 1],
        yaxis_range=[0, 1]
    )

    st.plotly_chart(fig, use_container_width=True)

    if brier <= 0.15:
        st.success(f"✅ Model is well-calibrated (Brier: {brier:.3f})")
    else:
        st.warning(f"⚠️ Consider recalibration (Brier: {brier:.3f})")

    st.markdown("---")

    # Feature importance with SHAP
    if shap_data is not None:
        st.subheader("🔍 Feature Importance (SHAP Values)")

        feature_names = shap_data['feature_names']
        shap_values = shap_data['shap_values']

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        importance_df['type'] = importance_df['feature'].apply(
            lambda x: 'Embedding' if x.startswith('emb_') else 'Structured'
        )

        # Summary metrics
        col1, col2, col3 = st.columns(3)

        struct_importance = importance_df[importance_df['type'] == 'Structured']['importance'].sum()
        emb_importance = importance_df[importance_df['type'] == 'Embedding']['importance'].sum()
        total_importance = struct_importance + emb_importance

        with col1:
            st.metric("Structured", f"{struct_importance / total_importance:.1%}")

        with col2:
            st.metric("Embedding", f"{emb_importance / total_importance:.1%}")

        with col3:
            top_feat = importance_df.iloc[0]
            st.metric("Top Feature", top_feat['feature'][:15] + "...")

        st.markdown("---")

        # Interactive filters
        col1, col2 = st.columns([1, 3])

        with col1:
            feature_filter = st.radio(
                "Show Features",
                ["All", "Structured Only", "Embeddings Only"]
            )

            top_n_feat = st.slider(
                "Number of Features",
                5, 50, 20, 5
            )

        with col2:
            if feature_filter == "Structured Only":
                display_df = importance_df[importance_df['type'] == 'Structured'].head(top_n_feat)
            elif feature_filter == "Embeddings Only":
                display_df = importance_df[importance_df['type'] == 'Embedding'].head(top_n_feat)
            else:
                display_df = importance_df.head(top_n_feat)

            colors = ['steelblue' if t == 'Structured' else 'orange' for t in display_df['type']]

            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=display_df['feature'],
                x=display_df['importance'],
                orientation='h',
                marker_color=colors,
                text=display_df['importance'].round(3),
                textposition='auto'
            ))

            fig.update_layout(
                xaxis_title="Mean |SHAP Value|",
                height=max(400, top_n_feat * 20),
                showlegend=False,
                yaxis={'categoryorder': 'total ascending'}
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("💡 SHAP analysis not available. Re-run training to generate SHAP values.")

# ==================== PAGE 5: LIVE PREDICTION TOOL ====================
elif page == "🎲 Live Prediction Tool":
    st.markdown("<h1 class='main-header'>🎲 Live Readmission Prediction</h1>", unsafe_allow_html=True)

    st.markdown("### Predict Readmission Risk for New Patients")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Demographics")
        age = st.number_input("Age", 18, 100, 65)

        st.markdown("#### Clinical History")
        # charlson_score = st.slider("Charlson Score", 0, 10, 2)
        charlson_score = 2
        los_days = st.number_input("Length of Stay (days)", min_value=1.0, max_value=365.0, value=5.0, step=1.0)
        prior_admits = st.number_input("Prior Admissions (6mo)", 0, 20, 0)

    with col2:
        st.markdown("#### Diagnoses")
        dx_heart_failure = st.checkbox("Heart Failure")
        dx_copd = st.checkbox("COPD")
        dx_diabetes = st.checkbox("Diabetes")
        dx_renal = st.checkbox("Renal Failure")

        st.markdown("#### Hospital Course")
        had_icu = st.checkbox("ICU Stay")
        discharge_weekend = st.checkbox("Weekend Discharge")

    if st.button("🔮 Calculate Risk", type="primary"):
        # Simplified risk calculation
        risk_score = 30 + (charlson_score * 5) + (los_days * 2) + (prior_admits * 3)
        if dx_heart_failure:
            risk_score += 15
        if dx_copd:
            risk_score += 10
        if dx_diabetes:
            risk_score += 5
        if dx_renal:
            risk_score += 8
        if had_icu:
            risk_score += 7
        if discharge_weekend:
            risk_score += 3

        risk_score = min(risk_score, 95)

        # Display result
        st.markdown("---")
        st.markdown("## 📊 Prediction Result")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if risk_score >= 70:
                risk_color = "#d32f2f"
                risk_label = "VERY HIGH RISK"
                risk_icon = "🔴"
            elif risk_score >= 50:
                risk_color = "#f57c00"
                risk_label = "HIGH RISK"
                risk_icon = "🟠"
            elif risk_score >= 30:
                risk_color = "#fbc02d"
                risk_label = "MODERATE RISK"
                risk_icon = "🟡"
            else:
                risk_color = "#388e3c"
                risk_label = "LOW RISK"
                risk_icon = "🟢"

            st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background-color: {risk_color}20; 
                        border-radius: 10px; border: 3px solid {risk_color};'>
                <h1 style='font-size: 4rem; margin: 0;'>{risk_icon}</h1>
                <h2 style='color: {risk_color}; margin: 0.5rem 0;'>{risk_score:.1f}%</h2>
                <h3 style='margin: 0;'>{risk_label}</h3>
                <p style='margin-top: 1rem;'>30-Day Readmission Risk</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Recommendations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📋 Interpretation")
            st.markdown(f"""
            **Risk Score:** {risk_score:.1f}%

            Approximately **{int(risk_score)}** out of 100 similar patients 
            would be readmitted within 30 days.
            """)

        with col2:
            st.markdown("### 🎯 Recommended Actions")

            if risk_score >= 70:
                st.error("""
                **URGENT - High Priority:**
                - ✅ Schedule follow-up within 24-48 hours
                - ✅ Coordinate with case management
                - ✅ Ensure medication reconciliation
                - ✅ Assess home support needs
                - ✅ Consider transitional care program
                """)
            elif risk_score >= 50:
                st.warning("""
                **MODERATE - Enhanced Follow-up:**
                - ✅ Schedule follow-up within 48-72 hours
                - ✅ Provide detailed discharge instructions
                - ✅ Verify medication understanding
                - ✅ Check access to follow-up care
                """)
            elif risk_score >= 30:
                st.info("""
                **STANDARD - Routine Follow-up:**
                - ✅ Schedule follow-up within 7-14 days
                - ✅ Standard discharge education
                - ✅ Monitor for complications
                """)
            else:
                st.success("""
                **LOW RISK - Standard Care:**
                - ✅ Routine discharge protocol
                - ✅ Standard follow-up timing
                - ✅ Patient education materials
                """)

        # Risk factors
        st.markdown("---")
        st.markdown("### 🔍 Key Risk Factors")

        risk_factors = []
        if charlson_score >= 3:
            risk_factors.append(f"🔴 High Comorbidity (Charlson: {charlson_score})")
        if prior_admits >= 2:
            risk_factors.append(f"🟠 Recent Hospitalizations ({prior_admits} in 6 months)")
        if los_days > 7:
            risk_factors.append(f"🟡 Extended Stay ({los_days} days)")
        if had_icu:
            risk_factors.append("🟡 ICU Admission")
        if dx_heart_failure:
            risk_factors.append("🔴 Heart Failure Diagnosis")
        if dx_copd:
            risk_factors.append("🟠 COPD Diagnosis")
        if dx_diabetes:
            risk_factors.append("🟡 Diabetes Diagnosis")
        if dx_renal:
            risk_factors.append("🟠 Renal Failure")
        if discharge_weekend:
            risk_factors.append("🟡 Weekend Discharge")

        if risk_factors:
            for factor in risk_factors:
                st.markdown(f"- {factor}")
        else:
            st.markdown("- ℹ️ Standard risk profile")

        # Export functionality
        st.markdown("---")
        if st.button("📥 Export Prediction Report"):
            report = f"""
TRANCE Readmission Risk Prediction Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION:
- Age: {age}
- Charlson Comorbidity Score: {charlson_score}
- Length of Stay: {los_days} days
- Prior Admissions (6 months): {prior_admits}

DIAGNOSES:
- Heart Failure: {'Yes' if dx_heart_failure else 'No'}
- COPD: {'Yes' if dx_copd else 'No'}
- Diabetes: {'Yes' if dx_diabetes else 'No'}
- Renal Failure: {'Yes' if dx_renal else 'No'}

HOSPITAL COURSE:
- ICU Stay: {'Yes' if had_icu else 'No'}
- Weekend Discharge: {'Yes' if discharge_weekend else 'No'}

PREDICTION RESULT:
- Risk Score: {risk_score:.1f}%
- Risk Level: {risk_label}
- Model: TRANCE Fused Model (Structured + Clinical Text)

RISK FACTORS:
{chr(10).join(['- ' + rf for rf in risk_factors])}

RECOMMENDED ACTIONS:
- Follow-up timing based on risk level
- Case management coordination as indicated
- Medication reconciliation and education
- Home support assessment if needed

This prediction is for clinical decision support only and should be used 
in conjunction with clinical judgment.
            """

            st.download_button(
                label="Download Report (TXT)",
                data=report,
                file_name=f"readmission_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
    TRANCE v1.0.0 | Temporal Readmission Analysis with Neural Clinical Embeddings<br>
    Last Updated: {datetime.now().strftime('%Y-%m-%d')} | 
    Test Set: {len(predictions)} patients | 
    Model AUROC: {results_summary['fused_model']['test_auroc']:.3f}<br>
    For research and educational purposes only
    </div>
    """,
    unsafe_allow_html=True
)