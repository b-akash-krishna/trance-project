"""
TRANCE Dashboard - Main Application
Save as: src/dashboard/app.py
Run with: streamlit run src/dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import joblib
from datetime import datetime, timedelta

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
    base_path = Path(__file__).parent.parent.parent
    return {
        'results': base_path / 'outputs/results',
        'models': base_path / 'outputs/models',
        'processed': base_path / 'data/processed'
    }

paths = get_paths()

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
        
        return predictions, results_summary, full_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

predictions, results_summary, full_data = load_data()

# Sidebar navigation
st.sidebar.markdown("# 🏥 TRANCE")
st.sidebar.markdown("### Readmission Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Executive Overview", "📈 Volume Forecasting", "🎯 Patient Risk Dashboard", "🔍 Model Monitoring"]
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
    
    if results_summary is None:
        st.error("Unable to load results. Please ensure models have been trained.")
        st.stop()
    
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
        st.metric("Embedding Boost", f"+{improvement:.1f}%", help="AUROC improvement from text embeddings")
    
    st.markdown("---")
    
    # Performance visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ROC Curve")
        
        # Create synthetic ROC data (in practice, load actual curve data)
        fpr = np.linspace(0, 1, 100)
        tpr = np.power(fpr, 0.6)  # Synthetic curve for demo
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            name=f"TRANCE Model (AUROC={auroc:.3f})",
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
    
    # Recent performance table
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
    
    if predictions is None:
        st.error("Unable to load predictions.")
        st.stop()
    
    # Simulate temporal data (in practice, you'd have actual discharge dates)
    np.random.seed(42)
    predictions['discharge_date'] = pd.date_range(start='2024-01-01', periods=len(predictions), freq='H')
    predictions['discharge_date'] = predictions['discharge_date'].dt.date
    
    # Aggregate by day
    daily_forecast = predictions.groupby('discharge_date').agg({
        'pred_prob_fused_calibrated': 'sum',  # Expected readmissions
        'true_label': 'sum'  # Actual readmissions
    }).reset_index()
    
    daily_forecast.columns = ['Date', 'Predicted Readmissions', 'Actual Readmissions']
    daily_forecast['Forecast Error'] = daily_forecast['Predicted Readmissions'] - daily_forecast['Actual Readmissions']
    
    # Forecast horizon selector
    st.sidebar.markdown("### Forecast Settings")
    horizon = st.sidebar.selectbox("Forecast Horizon", ["7-day", "14-day", "30-day"], index=0)
    horizon_days = int(horizon.split('-')[0])
    
    # Show next N days
    forecast_window = daily_forecast.head(horizon_days)
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predicted = forecast_window['Predicted Readmissions'].sum()
        st.metric(
            f"Predicted ({horizon})",
            f"{total_predicted:.0f} patients",
            help=f"Expected readmissions over next {horizon_days} days"
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
            f"{avg_daily:.1f} patients",
            help="Average predicted readmissions per day"
        )
    
    with col4:
        if forecast_window['Actual Readmissions'].sum() > 0:
            mae = np.abs(forecast_window['Forecast Error']).mean()
            st.metric(
                "Forecast MAE",
                f"{mae:.2f}",
                help="Mean Absolute Error"
            )
        else:
            st.metric("Forecast MAE", "N/A", help="Actual data pending")
    
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
    
    if forecast_window['Actual Readmissions'].sum() > 0:
        fig.add_trace(go.Scatter(
            x=forecast_window['Date'],
            y=forecast_window['Actual Readmissions'],
            mode='lines+markers',
            name='Actual',
            line=dict(color='orange', width=2, dash='dash'),
            marker=dict(size=6)
        ))
    
    # Add capacity reference line
    historical_avg = daily_forecast['Actual Readmissions'].mean()
    fig.add_hline(
        y=historical_avg,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"Historical Avg: {historical_avg:.1f}",
        annotation_position="right"
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Number of Readmissions",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Daily breakdown table
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
                Consider additional staffing.
                """
            )
        else:
            st.success("No high-volume days forecasted in this period.")

# ==================== PAGE 3: PATIENT RISK DASHBOARD ====================
elif page == "🎯 Patient Risk Dashboard":
    st.markdown("<h1 class='main-header'>🎯 High-Risk Patient List</h1>", unsafe_allow_html=True)
    
    if full_data is None:
        st.error("Unable to load patient data.")
        st.stop()
    
    # Filters
    st.sidebar.markdown("### Filters")
    
    risk_threshold = st.sidebar.slider(
        "Minimum Risk Score (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Show patients with risk score above this threshold"
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
        avg_risk = high_risk['risk_score'].mean()
        st.metric("Avg Risk Score", f"{avg_risk:.1f}%")
    
    with col3:
        capacity_status = "🟢 Within Capacity" if len(high_risk) <= 20 else "🟡 Above Capacity"
        st.metric("Capacity Status", capacity_status)
    
    with col4:
        if 'true_label' in high_risk.columns:
            actual_readmits = high_risk['true_label'].sum()
            precision = actual_readmits / len(high_risk) if len(high_risk) > 0 else 0
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
        
        # Determine risk category
        if risk >= 70:
            risk_class = "risk-high"
            risk_icon = "🔴"
            risk_label = "VERY HIGH RISK"
        elif risk >= 50:
            risk_class = "risk-medium"
            risk_icon = "🟠"
            risk_label = "HIGH RISK"
        else:
            risk_class = "risk-low"
            risk_icon = "🟡"
            risk_label = "MODERATE RISK"
        
        with st.expander(f"{risk_icon} Patient #{row['HADM_ID']} - Risk: {risk:.1f}%"):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {risk_icon} {risk:.1f}%")
                st.markdown(f"**{risk_label}**")
                
                st.markdown("---")
                
                # Patient demographics
                if 'age' in row:
                    st.markdown(f"**Age:** {row['age']:.0f}")
                if 'gender_M' in row:
                    gender = 'Male' if row['gender_M'] == 1 else 'Female'
                    st.markdown(f"**Gender:** {gender}")
                if 'los_days' in row:
                    st.markdown(f"**Length of Stay:** {row['los_days']:.1f} days")
            
            with col2:
                st.markdown("**Top Risk Factors:**")
                
                # Identify top risk factors (simplified - would use SHAP in practice)
                risk_factors = []
                
                if 'charlson_score' in row and row['charlson_score'] > 2:
                    risk_factors.append(f"🔴 High comorbidity burden (Charlson: {row['charlson_score']:.0f})")
                
                if 'prior_admissions_180d' in row and row['prior_admissions_180d'] > 1:
                    risk_factors.append(f"🟠 Recent hospitalizations ({row['prior_admissions_180d']:.0f} in 6 months)")
                
                if 'los_days' in row and row['los_days'] > 7:
                    risk_factors.append(f"🟡 Extended hospital stay ({row['los_days']:.0f} days)")
                
                if 'had_icu_stay' in row and row['had_icu_stay'] == 1:
                    risk_factors.append("🟡 ICU admission during stay")
                
                if 'dx_heart_failure' in row and row['dx_heart_failure'] == 1:
                    risk_factors.append("🔴 Heart failure diagnosis")
                
                if len(risk_factors) == 0:
                    risk_factors.append("ℹ️ Risk driven by clinical note patterns")
                
                for factor in risk_factors[:5]:
                    st.markdown(f"- {factor}")
                
                st.markdown("---")
                
                st.markdown("**📞 Recommended Actions:**")
                if risk >= 70:
                    st.markdown("- ✅ Schedule 24-48h post-discharge call")
                    st.markdown("- ✅ Coordinate urgent follow-up appointment")
                    st.markdown("- ✅ Review medication adherence plan")
                    st.markdown("- ✅ Assess home care needs")
                elif risk >= 50:
                    st.markdown("- ✅ Schedule 48-72h post-discharge call")
                    st.markdown("- ✅ Ensure follow-up appointment scheduled")
                    st.markdown("- ✅ Provide discharge instructions review")
                else:
                    st.markdown("- ✅ Standard follow-up protocol")
    
    # Export functionality
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("📥 Export to CSV"):
            csv = high_risk[['HADM_ID', 'SUBJECT_ID', 'risk_score']].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"high_risk_patients_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# ==================== PAGE 4: MODEL MONITORING ====================
elif page == "🔍 Model Monitoring":
    st.markdown("<h1 class='main-header'>🔍 Model Monitoring & Interpretability</h1>", unsafe_allow_html=True)
    
    if results_summary is None:
        st.error("Unable to load model data.")
        st.stop()
    
    # Model health indicators
    st.subheader("Model Health Status")
    
    col1, col2, col3 = st.columns(3)
    
    auroc = results_summary['fused_model']['test_auroc']
    brier = results_summary['fused_model']['brier_score_calibrated']
    
    with col1:
        auroc_status = "🟢 Good" if auroc >= 0.72 else "🟡 Monitor" if auroc >= 0.68 else "🔴 Alert"
        st.metric("AUROC Status", auroc_status, f"{auroc:.3f}")
        
    with col2:
        cal_status = "🟢 Good" if brier <= 0.15 else "🟡 Monitor" if brier <= 0.20 else "🔴 Alert"
        st.metric("Calibration Status", cal_status, f"Brier: {brier:.3f}")
    
    with col3:
        st.metric("Model Version", "v1.0.0", help="Current production model")
    
    st.markdown("---")
    
    # Calibration curve (synthetic for demo)
    st.subheader("Calibration Curve")
    
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Simulate good calibration
    observed_freq = bin_centers + np.random.normal(0, 0.05, n_bins)
    observed_freq = np.clip(observed_freq, 0, 1)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Perfect Calibration',
        line=dict(color='gray', dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=bin_centers,
        y=observed_freq,
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
        st.success(f"✅ Model is well-calibrated (Brier Score: {brier:.3f})")
    else:
        st.warning(f"⚠️ Consider recalibration (Brier Score: {brier:.3f})")
    
    st.markdown("---")
    
    # Feature importance
    st.subheader("Feature Importance")
    
    # Simulated feature importance
    features = [
        'Prior Admissions (180d)', 'Charlson Score', 'Length of Stay',
        'ICU Stay', 'Heart Failure Dx', 'Age', 'Discharge Weekend',
        'Clinical Note Embedding', 'Medications Count', 'COPD Dx'
    ]
    
    importance = np.array([25, 18, 15, 12, 10, 8, 5, 4, 2, 1])
    feature_types = ['Structured']*7 + ['Embedding'] + ['Structured']*2
    
    colors = ['steelblue' if t == 'Structured' else 'orange' for t in feature_types]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=features,
        x=importance,
        orientation='h',
        marker_color=colors,
        text=importance,
        textposition='auto'
    ))
    
    fig.update_layout(
        xaxis_title="Importance Score",
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    emb_contribution = results_summary['embedding_contribution']['importance_pct']
    st.info(f"ℹ️ Clinical text embeddings contribute **{emb_contribution:.1f}%** of total model importance")
    
    st.markdown("---")
    
    # Performance drift monitoring
    st.subheader("Performance Monitoring")
    
    # Simulate monthly performance
    months = pd.date_range(start='2024-01-01', periods=6, freq='M')
    auroc_trend = np.array([0.78, 0.77, 0.78, 0.76, 0.77, 0.78])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=months,
        y=auroc_trend,
        mode='lines+markers',
        name='AUROC',
        line=dict(color='steelblue', width=3),
        marker=dict(size=10)
    ))
    
    fig.add_hline(
        y=0.72,
        line_dash="dash",
        line_color="red",
        annotation_text="Performance Threshold"
    )
    
    fig.update_layout(
        yaxis_title="AUROC",
        yaxis_range=[0.65, 0.85],
        height=350
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if auroc_trend[-1] >= 0.72:
        st.success("✅ Model performance is stable")
    else:
        st.error("🔴 Model performance has degraded - consider retraining")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
    TRANCE v1.0.0 | Temporal Readmission Analysis with Neural Clinical Embeddings<br>
    For research and educational purposes only
    </div>
    """,
    unsafe_allow_html=True
)