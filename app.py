# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import os

@st.cache_data
def load_data():
    return pd.read_csv("./data/orders_bike_cleaned.csv")

df_full = load_data()

st.set_page_config(page_title="Bike Sales Dashboard", layout="wide", initial_sidebar_state="expanded")

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
decision_boundary_img = os.path.join(PRECOMP_DIR, "decision_boundary_logistic.png")

# Sidebar
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Select section", ["Overview", "Univariate", "Bivariate", "Multivariate", "Models"])

st.title("📊 Bike Sales Dashboard")

# ---------- Overview ----------
if tab == "Overview":
    st.header("Overview & Key Metrics")
    if summary is not None:
        cols = st.columns(3)
        try:
            total_orders = int(summary.loc[summary['Metric']=='Total Orders','Value'].values[0])
            total_revenue = float(summary.loc[summary['Metric']=='Total Revenue','Value'].values[0])
            avg_price = float(summary.loc[summary['Metric']=='Average Price','Value'].values[0])
        except Exception:
            # fallback if saved differently
            total_orders = summary['Value'][0]
            total_revenue = summary['Value'][1]
            avg_price = summary['Value'][2]
        cols[0].metric("Total Orders", f"{total_orders:,}")
        cols[1].metric("Total Revenue", f"${total_revenue:,.2f}")
        cols[2].metric("Average Price", f"${avg_price:,.2f}")
    else:
        st.info("Summary stats not found. Run precompute.py to generate precomputed files.")

    st.markdown("**Data Sources:** `data/orders_bike_cleaned.csv` → precomputed files in `precomputed/`")
    st.write("Use the left sidebar to navigate to EDA and Models.")

# ---------- Univariate ----------
elif tab == "Univariate":
    st.header("Univariate Analysis (distributions)")
    st.write("Expand the plots you want to view to avoid long initial load.")

    # numeric cols list
    numeric_cols = load_csv(os.path.join(PRECOMP_DIR, "numeric_cols.csv"))
    cat_cols = load_csv(os.path.join(PRECOMP_DIR, "cat_cols.csv"))

    df_path = "./data/orders_bike_cleaned.csv"
    if os.path.exists(df_path):
        df_full = pd.read_csv(df_path)
    else:
        df_full = None

    with st.expander("Numeric distributions (histograms)"):
        if df_full is None:
            st.warning("Raw dataset not found at ./data/orders_bike_cleaned.csv — run precompute or place dataset there.")
        else:
            # show a selectbox of numeric columns
            if numeric_cols is not None and not numeric_cols.empty:
                options = numeric_cols['numeric_cols'].tolist()
            else:
                options = df_full.select_dtypes(include=['number']).columns.tolist()
            col_choice = st.selectbox("Choose numeric column", options)
            fig = px.histogram(df_full, x=col_choice, nbins=40, marginal="box", title=f"Distribution: {col_choice}")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Categorical counts"):
        if df_full is None:
            st.warning("Raw dataset not found.")
        else:
            if cat_cols is not None and not cat_cols.empty:
                cat_options = cat_cols['cat_cols'].tolist()
            else:
                cat_options = df_full.select_dtypes(include=['object']).columns.tolist()
            cat_choice = st.selectbox("Choose categorical column", cat_options, index=0)
            # limit top 20
            vc = df_full[cat_choice].value_counts().nlargest(20).reset_index()
            vc.columns = [cat_choice, 'count']
            fig = px.bar(vc, x=cat_choice, y='count', title=f"Top categories (up to 20) for {cat_choice}")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Bivariate ----------
elif tab == "Bivariate":
    st.header("Bivariate Analysis")
    with st.expander("Correlation heatmap (numeric)"):
        if numeric_corr is None:
            st.warning("numeric_corr.csv not found. Run precompute.py")
        else:
            # numeric_corr is square matrix (index in first column)
            fig = go.Figure(data=go.Heatmap(
                z=numeric_corr.values,
                x=numeric_corr.columns.tolist(),
                y=numeric_corr.index.tolist(),
                colorscale="RdBu",
                zmid=0
            ))
            fig.update_layout(title="Correlation heatmap (numeric)")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Sales by Category1 / Category2"):
        if category1_sales is not None:
            fig1 = px.bar(category1_sales.sort_values('total_sales', ascending=False), x='category1', y='total_sales', title='Total Sales by Category1')
            st.plotly_chart(fig1, use_container_width=True)
        if category2_sales is not None:
            fig2 = px.bar(category2_sales.sort_values('total_sales', ascending=False), x='category2', y='total_sales', title='Total Sales by Category2')
            st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Sales by Category1 × Category2 (stacked)"):
        if comb_sales is not None:
            pivot = comb_sales.pivot(index='category1', columns='category2', values='total_sales').fillna(0)
            fig = px.bar(pivot, title='Sales by Category1 & Category2 (stacked)')
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Price vs Quantity scatter"):
        if df_full is None:
            st.warning("Raw dataset not found.")
        else:
            if 'price' in df_full.columns and 'quantity' in df_full.columns:
                fig = px.scatter(df_full, x='price', y='quantity', trendline='ols', title='Price vs Quantity')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("price/quantity columns missing in raw dataset.")

    with st.expander("Monthly sales trend"):
        if monthly_sales is not None:
            fig = px.line(monthly_sales, x='Month', y='Total Sales', markers=True, title='Monthly Sales Trend')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("monthly_sales.csv not found")

# ---------- Multivariate ----------
elif tab == "Multivariate":
    st.header("Multivariate Analysis")

    with st.expander("Average Price: Category1 × Category2 (heatmap)"):
        if pivot_cat is None:
            st.warning("pivot_cat.csv not found.")
        else:
            # pivot_cat index is category1, columns category2
            fig = go.Figure(data=go.Heatmap(
                z=pivot_cat.values,
                x=pivot_cat.columns.tolist(),
                y=pivot_cat.index.tolist(),
                colorscale='Blues',
                hoverongaps=False
            ))
            fig.update_layout(title="Average Price by Category1 × Category2")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Average Price: Category1 × Frame (heatmap)"):
        if pivot_frame is None:
            st.warning("pivot_frame.csv not found.")
        else:
            fig = go.Figure(data=go.Heatmap(
                z=pivot_frame.values,
                x=pivot_frame.columns.tolist(),
                y=pivot_frame.index.tolist(),
                colorscale='Greens',
                hoverongaps=False
            ))
            fig.update_layout(title="Average Price by Category1 × Frame")
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Price distribution by Category1 (boxplot)"):
        if df_full is None:
            st.warning("Raw dataset not found.")
        else:
            if 'category1' in df_full.columns and 'price' in df_full.columns:
                fig = px.box(df_full, x='category1', y='price', title='Price Distribution by Category1')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("columns missing")

    with st.expander("Price by Frame and Category1 (boxplot)"):
        if df_full is None:
            st.warning("Raw dataset not found.")
        else:
            if 'frame' in df_full.columns and 'price' in df_full.columns and 'category1' in df_full.columns:
                # show an example subset to keep plot readable
                subset = df_full.copy()
                fig = px.box(subset, x='frame', y='price', color='category1', title='Price by Frame and Category1')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("columns missing")

    with st.expander("Monthly orders by Category1 (multi-line)"):
        if monthly_orders is None:
            st.warning("monthly_orders.csv not found.")
        else:
            # pivot to wide
            wide = monthly_orders.pivot(index='month', columns='category1', values='order_count').fillna(0)
            fig = go.Figure()
            for col in wide.columns:
                fig.add_trace(go.Scatter(x=wide.index, y=wide[col], mode='lines+markers', name=str(col)))
            fig.update_layout(title='Monthly Orders by Category1', xaxis_title='Month', yaxis_title='Order count')
            st.plotly_chart(fig, use_container_width=True)

# ---------- Models ----------
elif tab == "Models":
    st.header("Model Comparison & ROC")

    if model_results is None:
        st.warning("model_results.csv not found. Run precompute.py to generate models and metrics.")
    else:
        with st.expander("Model comparison table"):
            st.dataframe(model_results.style.format({"Accuracy":"{:.3f}", "Precision":"{:.3f}", "Recall":"{:.3f}", "F1":"{:.3f}", "AUC_macro":"{:.3f}"}))

        with st.expander("Accuracy bar chart"):
            fig = px.bar(model_results.sort_values('Accuracy', ascending=False), x='Model', y='Accuracy', title='Model Accuracy')
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("ROC curves (per-class averaged AUC if available)"):
            if roc_data is None:
                st.info("ROC data not available.")
            else:
                # show each model's per-class ROC as small facets
                for model_name, d in roc_data.items():
                    st.markdown(f"**{model_name}** (per-class curves)")
                    fpr_list = d.get('fpr', [])
                    tpr_list = d.get('tpr', [])
                    per_class_auc = d.get('per_class_auc', [])
                    classes = d.get('classes', [])
                    fig = go.Figure()
                    for i, (fpr, tpr) in enumerate(zip(fpr_list, tpr_list)):
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"Class {classes[i]} (AUC={per_class_auc[i]:.3f})"))
                    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
                    fig.update_layout(title=f'ROC Curves - {model_name}', xaxis_title='FPR', yaxis_title='TPR')
                    st.plotly_chart(fig, use_container_width=True)

        with st.expander("Decision boundary image (Logistic)"):
            if os.path.exists(decision_boundary_img):
                st.image(decision_boundary_img, use_column_width=True)
            else:
                st.info("Decision boundary image not found (precompute.py creates it).")

# ---------- Footer ----------
st.markdown("---")
st.markdown("Dashboard built from precomputed CSVs. To update values re-run `precompute.py` and restart this Streamlit app.")
