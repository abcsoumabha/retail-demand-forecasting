import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error
import os

st.set_page_config(page_title="Hierarchical Demand Forecasting", layout="wide")
st.title("📊 Hierarchical Probabilistic Sales Forecasting")

st.markdown("""
Visualizing uncertainty in retail sales using LightGBM Quantile Regression and Bottom-Up Reconciliation.

**Implementation Notes:**
*   **Target:** Forecasting observed *sales* (censored by stockouts), not latent demand.
*   **Evaluation:** 7-day Seasonal Naive baseline included for WRMSSE/RMSE comparison.
*   **Reconciliation:** Forecasts aggregated from Item -> Department -> Store levels.
*   **⚠️ VIEW MODE: HISTORICAL BACKTEST.** *Actuals are shown alongside 28-day rolling-origin forecasts for model validation.*
""")
st.divider()

@st.cache_data
def load_data():
    # THIS IS THE CRITICAL FIX
    file_path = os.path.join("data", "demo_forecasts.parquet")
    
    if not os.path.exists(file_path):
        # We are also updating the error message so we know if this new code runs!
        st.error(f"Cannot find {file_path}. Make sure your 'data' folder contains 'demo_forecasts.parquet'.")
        return pd.DataFrame()
    return pd.read_parquet(file_path)

with st.spinner("Loading hierarchical predictions..."):
    df = load_data()

if not df.empty:
    st.sidebar.header("Hierarchy Navigation")
    
    levels = ['Store', 'Department', 'Item']
    selected_level = st.sidebar.radio("1. Select Hierarchy Level", levels)
    
    level_df = df[df['Level'] == selected_level]
    entities = level_df['Entity'].unique()
    selected_entity = st.sidebar.selectbox("2. Select Entity", entities)
    
    plot_df = level_df[level_df['Entity'] == selected_entity].sort_values('d_num').dropna(subset=['snaive'])

    actuals = plot_df['sales']
    lgbm_rmse = np.sqrt(mean_squared_error(actuals, plot_df['pred_q50']))
    snaive_rmse = np.sqrt(mean_squared_error(actuals, plot_df['snaive']))
    
    within_interval = ((actuals >= plot_df['pred_q5']) & (actuals <= plot_df['pred_q95']))
    coverage_pct = within_interval.mean() * 100

    st.subheader(f"Backtest Analysis: {selected_entity} ({selected_level} Level)")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("LightGBM RMSE (Median)", f"{lgbm_rmse:.2f}", delta=f"{snaive_rmse - lgbm_rmse:.2f} vs Naive", delta_color="normal")
    col2.metric("Seasonal Naive RMSE", f"{snaive_rmse:.2f}")
    col3.metric("90% Interval Coverage", f"{coverage_pct:.1f}%", help="Target is ~90%.")
    
    st.write("") 

    x_vals = plot_df['d_num'].astype(float).tolist()
    y_actual = actuals.astype(float).tolist()
    y_q5 = plot_df['pred_q5'].astype(float).tolist()
    y_q50 = plot_df['pred_q50'].astype(float).tolist()
    y_q95 = plot_df['pred_q95'].astype(float).tolist()
    y_snaive = plot_df['snaive'].astype(float).tolist()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_vals + x_vals[::-1],
        y=y_q95 + y_q5[::-1],
        fill='toself',
        fillcolor='rgba(0, 176, 246, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% Confidence Interval'
    ))

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_snaive, mode='lines', name='S-Naive Baseline', line=dict(color='orange', width=2, dash='dot')
    ))

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_q50, mode='lines', name='Median Forecast', line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_actual, mode='lines+markers', name='Actual Sales', line=dict(color='black', width=2), marker=dict(size=4)
    ))

    fig.update_layout(xaxis_title="Day (d_num)", yaxis_title="Units Sold", hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    st.plotly_chart(fig, width='stretch')
