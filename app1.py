import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

@st.cache_data
def load_data():
    return pd.read_csv("./data/orders_bike_cleaned.csv")

df_full = load_data()

st.set_page_config(page_title="Bike Sales Dashboard", layout="wide", initial_sidebar_state="collapsed")

PRECOMP_DIR = "./precomputed"
MODELS_DIR = "./models"

# ---------- Helpers ----------
@st.cache_data
def load_csv(path, index_col=None):
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=index_col)
        except Exception:
            return pd.read_csv(path)
    else:
        return None

@st.cache_resource
def load_pickle(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None

# Load precomputed files
summary = load_csv(os.path.join(PRECOMP_DIR, "summary_stats.csv"))
monthly_sales = load_csv(os.path.join(PRECOMP_DIR, "monthly_sales.csv"))
category1_sales = load_csv(os.path.join(PRECOMP_DIR, "category1_sales.csv"))
category2_sales = load_csv(os.path.join(PRECOMP_DIR, "category2_sales.csv"))
comb_sales = load_csv(os.path.join(PRECOMP_DIR, "category_combination_sales.csv"))
numeric_corr = load_csv(os.path.join(PRECOMP_DIR, "numeric_corr.csv"), index_col=0)
pivot_cat = load_csv(os.path.join(PRECOMP_DIR, "pivot_cat.csv"), index_col=0)
pivot_frame = load_csv(os.path.join(PRECOMP_DIR, "pivot_frame.csv"), index_col=0)
monthly_orders = load_csv(os.path.join(PRECOMP_DIR, "monthly_orders.csv"))
model_results = load_csv(os.path.join(PRECOMP_DIR, "model_results.csv"))
roc_data = load_pickle(os.path.join(PRECOMP_DIR, "roc_data.pkl"))
numeric_cols = load_csv(os.path.join(PRECOMP_DIR, "numeric_cols.csv"))
cat_cols = load_csv(os.path.join(PRECOMP_DIR, "cat_cols.csv"))

# ---------- Dashboard Header ----------
st.title("📊 Bike Sales Analytics Dashboard")

# ---------- Key Metrics Row ----------
st.subheader("📈 Key Performance Indicators")
if summary is not None:
    cols = st.columns(6)
    try:
        total_orders = int(summary.loc[summary['Metric']=='Total Orders','Value'].values[0])
        total_revenue = float(summary.loc[summary['Metric']=='Total Revenue','Value'].values[0])
        avg_price = float(summary.loc[summary['Metric']=='Average Price','Value'].values[0])
    except Exception:
        total_orders = summary['Value'][0]
        total_revenue = summary['Value'][1]
        avg_price = summary['Value'][2]
    
    cols[0].metric("📦 Total Orders", f"{total_orders:,}")
    cols[1].metric("💰 Total Revenue", f"${total_revenue:,.2f}")
    cols[2].metric("💵 Average Price", f"${avg_price:,.2f}")
    
    # Calculate additional metrics
    if df_full is not None:
        avg_quantity = df_full['quantity'].mean() if 'quantity' in df_full.columns else 0
        unique_products = df_full['product_line'].nunique() if 'product_line' in df_full.columns else 0
        top_category = df_full['category1'].mode()[0] if 'category1' in df_full.columns else "N/A"
        
        cols[3].metric("📊 Avg Quantity", f"{avg_quantity:.1f}")
        cols[4].metric("🚲 Unique Products", f"{unique_products}")
        cols[5].metric("🏆 Top Category", top_category)

st.markdown("---")

# ---------- Main Dashboard Grid ----------
# Row 1: Univariate Analysis (4 charts)
st.subheader("🔍 Univariate Analysis")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Price Distribution**")
    if df_full is not None and 'price' in df_full.columns:
        fig = px.histogram(df_full, x='price', nbins=30, title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Quantity Distribution**")
    if df_full is not None and 'quantity' in df_full.columns:
        fig = px.histogram(df_full, x='quantity', nbins=20, title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("**Category1 Counts**")
    if df_full is not None and 'category1' in df_full.columns:
        vc = df_full['category1'].value_counts().head(10)
        fig = px.bar(x=vc.index, y=vc.values, title="")
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("**Frame Type Distribution**")
    if df_full is not None and 'frame' in df_full.columns:
        vc = df_full['frame'].value_counts().head(8)
        fig = px.pie(values=vc.values, names=vc.index, title="")
        fig.update_layout(height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 2: Bivariate Analysis (4 charts)
st.subheader("🔗 Bivariate Analysis")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Monthly Sales Trend**")
    if monthly_sales is not None:
        fig = px.line(monthly_sales, x='Month', y='Total Sales', markers=True, title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Sales by Category1**")
    if category1_sales is not None:
        top_cats = category1_sales.sort_values('total_sales', ascending=False).head(8)
        fig = px.bar(top_cats, x='category1', y='total_sales', title="")
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("**Price vs Quantity**")
    if df_full is not None and 'price' in df_full.columns and 'quantity' in df_full.columns:
        # Sample data for performance
        sample_df = df_full.sample(min(1000, len(df_full)))
        fig = px.scatter(sample_df, x='price', y='quantity', opacity=0.6, title="")
        fig.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("**Category2 Performance**")
    if category2_sales is not None:
        top_cat2 = category2_sales.sort_values('total_sales', ascending=False).head(6)
        fig = px.bar(top_cat2, x='category2', y='total_sales', title="")
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 3: Multivariate Analysis (4 charts)
st.subheader("🎯 Multivariate Analysis")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**Price by Category1 (Box Plot)**")
    if df_full is not None and 'category1' in df_full.columns and 'price' in df_full.columns:
        fig = px.box(df_full, x='category1', y='price', title="")
        fig.update_layout(height=300, showlegend=False, xaxis_tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("**Correlation Heatmap**")
    if numeric_corr is not None:
        # Show top correlations only
        corr_subset = numeric_corr.iloc[:6, :6] if numeric_corr.shape[0] > 6 else numeric_corr
        fig = go.Figure(data=go.Heatmap(
            z=corr_subset.values,
            x=corr_subset.columns.tolist(),
            y=corr_subset.index.tolist(),
            colorscale="RdBu",
            zmid=0
        ))
        fig.update_layout(height=300, title="")
        st.plotly_chart(fig, use_container_width=True)

with col3:
    st.markdown("**Monthly Orders by Category1**")
    if monthly_orders is not None:
        # Show top 3 categories only
        top_categories = monthly_orders.groupby('category1')['order_count'].sum().nlargest(3).index
        filtered_data = monthly_orders[monthly_orders['category1'].isin(top_categories)]
        fig = px.line(filtered_data, x='month', y='order_count', color='category1', title="")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

with col4:
    st.markdown("**Price Heatmap: Category1 × Frame**")
    if pivot_frame is not None:
        # Show subset if too large
        subset = pivot_frame.iloc[:6, :6] if pivot_frame.shape[0] > 6 else pivot_frame
        fig = go.Figure(data=go.Heatmap(
            z=subset.values,
            x=subset.columns.tolist(),
            y=subset.index.tolist(),
            colorscale='Viridis'
        ))
        fig.update_layout(height=300, title="")
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Row 4: Model Performance (2 large charts)
st.subheader("🤖 Model Performance & Predictions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Model Comparison**")
    if model_results is not None:
        fig = px.bar(model_results.sort_values('Accuracy', ascending=True), 
                     x='Accuracy', y='Model', orientation='h', title="")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show model metrics table
        st.markdown("**Detailed Metrics**")
        formatted_results = model_results.style.format({
            "Accuracy": "{:.3f}", "Precision": "{:.3f}", 
            "Recall": "{:.3f}", "F1": "{:.3f}", "AUC_macro": "{:.3f}"
        })
        st.dataframe(formatted_results, use_container_width=True)

with col2:
    st.markdown("**ROC Curves (Best Model)**")
    if roc_data is not None and model_results is not None:
        # Get best model
        best_model = model_results.loc[model_results['Accuracy'].idxmax(), 'Model']
        if best_model in roc_data:
            d = roc_data[best_model]
            fpr_list = d.get('fpr', [])
            tpr_list = d.get('tpr', [])
            per_class_auc = d.get('per_class_auc', [])
            classes = d.get('classes', [])
            
            fig = go.Figure()
            for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
                if i < 3:  # Show only first 3 classes for clarity
                    fig.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode='lines', 
                        name=f"Class {classes[i]} (AUC={per_class_auc[i]:.3f})"
                    ))
            fig.add_trace(go.Scatter(
                x=[0,1], y=[0,1], mode='lines', 
                line=dict(dash='dash'), name='Random'
            ))
            fig.update_layout(
                height=400, title=f'ROC Curves - {best_model}',
                xaxis_title='False Positive Rate',
                yaxis_title='True Positive Rate'
            )
            st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.markdown("**📊 Dashboard Overview:** Comprehensive bike sales analytics with univariate, bivariate, multivariate analysis, and ML model performance")
st.markdown("*Data refreshed from precomputed files. Re-run `precompute.py` to update visualizations.*")