# precompute.py
import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
DATA_PATH = "./data/orders_bike_cleaned.csv"   # your CSV path
PRECOMP_DIR = "./precomputed"
MODELS_DIR = "./models"
os.makedirs(PRECOMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- LOAD ----------
print("Loading data:", DATA_PATH)
df = pd.read_csv(DATA_PATH)

# Try to normalize common column names (some notebooks used order.date vs order_date)
if 'order.date' in df.columns and 'order_date' not in df.columns:
    df = df.rename(columns={'order.date': 'order_date'})
if 'order_date' not in df.columns and 'order.date' not in df.columns:
    # proceed, but some operations rely on order_date — we'll handle missing gracefully
    pass

# ---------- BASIC CLEANUP ----------
# ensure datetime
if 'order_date' in df.columns:
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['month'] = df['order_date'].dt.to_period('M').astype(str)
else:
    # fallback: try to create a dummy month column if not present
    df['month'] = "unknown"

# ensure numeric for quantity and price
df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce').fillna(0)
df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)

# create total_sales
df['total_sales'] = df['quantity'] * df['price']

# ---------- SUMMARY STATS ----------
summary = pd.DataFrame({
    'Metric': ['Total Orders', 'Total Revenue', 'Average Price'],
    'Value': [int(len(df)), float(df['total_sales'].sum()), float(df['price'].mean())]
})
summary.to_csv(os.path.join(PRECOMP_DIR, "summary_stats.csv"), index=False)
print("Saved summary_stats.csv")

# ---------- MONTHLY SALES ----------
monthly_sales = df.groupby('month', sort=True)['total_sales'].sum().reset_index().rename(columns={'total_sales':'Total Sales', 'month':'Month'})
monthly_sales.to_csv(os.path.join(PRECOMP_DIR, "monthly_sales.csv"), index=False)
print("Saved monthly_sales.csv")

# ---------- CATEGORY SALES ----------
if 'category1' in df.columns:
    cat1 = df.groupby('category1')['total_sales'].sum().reset_index().rename(columns={'total_sales':'total_sales'})
    cat1.to_csv(os.path.join(PRECOMP_DIR, "category1_sales.csv"), index=False)
    print("Saved category1_sales.csv")
if 'category2' in df.columns:
    cat2 = df.groupby('category2')['total_sales'].sum().reset_index().rename(columns={'total_sales':'total_sales'})
    cat2.to_csv(os.path.join(PRECOMP_DIR, "category2_sales.csv"), index=False)
    print("Saved category2_sales.csv")

# ---------- CATEGORY COMBINATION ----------
if 'category1' in df.columns and 'category2' in df.columns:
    comb = df.groupby(['category1','category2'])['total_sales'].sum().reset_index().rename(columns={'total_sales':'total_sales'})
    comb.to_csv(os.path.join(PRECOMP_DIR, "category_combination_sales.csv"), index=False)
    print("Saved category_combination_sales.csv")

# ---------- CORRELATION MATRIX (numeric) ----------
numeric_df = df.select_dtypes(include=['number']).copy()
corr = numeric_df.corr()
corr.to_csv(os.path.join(PRECOMP_DIR, "numeric_corr.csv"))
print("Saved numeric_corr.csv")

# ---------- PIVOTS (heatmaps) ----------
if 'category1' in df.columns and 'category2' in df.columns and 'price' in df.columns:
    pivot_cat = df.pivot_table(index='category1', columns='category2', values='price', aggfunc='mean')
    pivot_cat.to_csv(os.path.join(PRECOMP_DIR, "pivot_cat.csv"))
    print("Saved pivot_cat.csv")
if 'category1' in df.columns and 'frame' in df.columns and 'price' in df.columns:
    pivot_frame = df.pivot_table(index='category1', columns='frame', values='price', aggfunc='mean')
    pivot_frame.to_csv(os.path.join(PRECOMP_DIR, "pivot_frame.csv"))
    print("Saved pivot_frame.csv")

# ---------- MONTHLY ORDERS (long) ----------
if 'month' in df.columns and 'category1' in df.columns:
    monthly_orders = df.groupby(['month','category1']).size().reset_index(name='order_count')
    monthly_orders.to_csv(os.path.join(PRECOMP_DIR, "monthly_orders.csv"), index=False)
    print("Saved monthly_orders.csv")

# ---------- UNIVARIATE DETAILS (optionally save list of numeric / categorical cols) ----------
numeric_cols = numeric_df.columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
pd.DataFrame({'numeric_cols': numeric_cols}).to_csv(os.path.join(PRECOMP_DIR, "numeric_cols.csv"), index=False)
pd.DataFrame({'cat_cols': cat_cols}).to_csv(os.path.join(PRECOMP_DIR, "cat_cols.csv"), index=False)

# ---------- MODEL TRAINING & METRICS (optional but included) ----------
# We'll train multiple classifiers to produce model_results.csv and ROC data.
# Target: category1 (encoded). Features used: price, frame_encoded (as in your notebook).
if 'category1' in df.columns and 'frame' in df.columns and 'price' in df.columns:
    # encode labels
    le_cat1 = LabelEncoder().fit(df['category1'].astype(str))
    le_frame = LabelEncoder().fit(df['frame'].astype(str))
    df['category1_encoded'] = le_cat1.transform(df['category1'].astype(str))
    df['frame_encoded'] = le_frame.transform(df['frame'].astype(str))

    X = df[['price','frame_encoded']].values
    y = df['category1_encoded'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # classifiers to train
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }

    results = []
    roc_data = {}  # will store per-model per-class fpr/tpr
    for name, mdl in models.items():
        print("Training:", name)
        try:
            mdl.fit(X_train_s, y_train)
        except Exception as e:
            print(f"Could not train {name}: {e}")
            continue

        # predictions & metrics
        y_pred = mdl.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # try compute multiclass AUC (OVR) if predict_proba exists
        auc_val = np.nan
        fpr_list = []; tpr_list = []
        if hasattr(mdl, "predict_proba"):
            y_score = mdl.predict_proba(X_test_s)  # shape (n_samples, n_classes)
            try:
                # Binarize y_test
                classes = np.unique(y_train)
                y_test_bin = label_binarize(y_test, classes=classes)
                # compute per-class ROC curves and overall macro AUC
                per_class_auc = []
                for i in range(y_test_bin.shape[1]):
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                    fpr_list.append(fpr); tpr_list.append(tpr)
                    per_class_auc.append(auc(fpr, tpr))
                auc_val = float(np.mean(per_class_auc))
                roc_data[name] = {'fpr': fpr_list, 'tpr': tpr_list, 'per_class_auc': per_class_auc, 'classes': classes.tolist()}
            except Exception as e:
                print("Could not compute ROC for", name, e)
        else:
            try:
                # some models support decision_function
                if hasattr(mdl, "decision_function"):
                    y_score = mdl.decision_function(X_test_s)
                    # For binary vs multiclass handling becomes more complex; skip for robust demo.
            except Exception as e:
                pass

        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1,
            'AUC_macro': auc_val
        })

        # Save model pickle
        with open(os.path.join(MODELS_DIR, f"{name.lower().replace(' ','_')}.pkl"), "wb") as f:
            pickle.dump(mdl, f)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(PRECOMP_DIR, "model_results.csv"), index=False)
    print("Saved model_results.csv")

    # save roc_data (pickle)
    with open(os.path.join(PRECOMP_DIR, "roc_data.pkl"), "wb") as f:
        pickle.dump(roc_data, f)
    print("Saved roc_data.pkl")

    # save scaler & encoders
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(MODELS_DIR, "label_encoders.pkl"), "wb") as f:
        pickle.dump({'category1': le_cat1, 'frame': le_frame}, f)
    print("Saved scaler.pkl and label_encoders.pkl")

    # ---------- Decision boundary PNG (Logistic) ----------
    # We'll visualize predicted class across a grid for logistic regression.
    if 'Logistic Regression' in models:
        try:
            log = models['Logistic Regression']
            # grid over price and frame_encoded range
            price_min, price_max = X[:,0].min(), X[:,0].max()
            frame_min, frame_max = X[:,1].min(), X[:,1].max()
            xx, yy = np.meshgrid(np.linspace(price_min-100, price_max+100, 300),
                                 np.linspace(frame_min-1, frame_max+1, 200))
            grid = np.c_[xx.ravel(), yy.ravel()]
            grid_s = scaler.transform(grid)
            preds_grid = log.predict(grid_s).reshape(xx.shape)

            plt.figure(figsize=(10,6))
            plt.contourf(xx, yy, preds_grid, alpha=0.3, cmap='tab10')
            # scatter original points
            plt.scatter(X[:,0], X[:,1], c=y, cmap='tab10', edgecolor='k', s=30)
            plt.xlabel('Price')
            plt.ylabel('Frame encoded')
            plt.title('Decision Regions (Logistic Regression)')
            out_png = os.path.join(PRECOMP_DIR, "decision_boundary_logistic.png")
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()
            print("Saved decision boundary image:", out_png)
        except Exception as e:
            print("Could not make decision boundary:", e)

else:
    print("Skipping model precompute: missing columns (category1/frame/price).")

print(" Precompute finished. Files saved in:", PRECOMP_DIR, "and models in", MODELS_DIR)
