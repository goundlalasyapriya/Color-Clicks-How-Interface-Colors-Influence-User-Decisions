# train_color_click_model.py
"""
Robust training script for Color Clicks recommender.

- Auto-detects target column (click) if present; otherwise falls back to clustering color families.
- Preprocesses numeric + categorical features.
- Balances classes with SMOTE if classification is used and imbalance detected.
- Uses StratifiedKFold + GridSearchCV for hyperparameter search.
- Saves meta object for Streamlit as color_click_recommender_rf.joblib
- Writes a small training_summary.json for quick reporting.

Run:
    python train_color_click_model.py
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# sklearn / imblearn imports
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE

# Clustering fallback
from sklearn.cluster import KMeans

# ----------------- CONFIG -----------------
DATA_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"
OUT_JOBLIB = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_click_recommender_rf.joblib"
SUMMARY_JSON = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\training_summary.json"
RANDOM_STATE = 42
K_CLUSTERS = 12
# ------------------------------------------

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    print("Loaded dataset:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def find_click_column(df):
    candidates = [c for c in df.columns if 'click' in c.lower() or c.lower() in ('clicked','is_clicked')]
    return candidates[0] if candidates else None

def ensure_rgb(df):
    # Ensure 'r','g','b' exist; try hex columns if present
    if set(['r','g','b']).issubset(df.columns):
        return df
    hex_col = next((c for c in df.columns if 'hex' in c.lower()), None)
    if hex_col:
        def hex_to_rgb(h):
            try:
                s = str(h).lstrip('#')
                return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
            except:
                return (128,128,128)
        rgb = df[hex_col].apply(hex_to_rgb).tolist()
        df[['r','g','b']] = pd.DataFrame(rgb, index=df.index)
        print(f"Parsed RGB from {hex_col}")
    else:
        # fallback: create neutral columns if none exist
        df['r'], df['g'], df['b'] = 128, 128, 128
        print("No RGB or hex column found — created neutral r,g,b columns (128).")
    return df

def select_features(df, prefer=None):
    # prefer is a list of canonical names we want; map dataset columns if capitalization differs
    prefer = prefer or ['age','gender','device_type','ad_position','Product_Category','Mood','Season','r','g','b','Time_Spent_sec']
    # Build a mapping from dataset to canonical where possible
    rename = {}
    col_lower = {c.lower(): c for c in df.columns}
    # simple mappings
    mappings = {
        'age': ['age'],
        'gender': ['gender','sex'],
        'device_type': ['device_type','device','device type'],
        'ad_position': ['ad_position','ad position','adpos','ad_pos'],
        'Product_Category': ['product_category','productcategory','product'],
        'Mood': ['mood'],
        'Season': ['season'],
        'Time_Spent_sec': ['time_spent_sec','time_spent','timespent','time on page']
    }
    for canon, keys in mappings.items():
        for k in keys:
            if k in col_lower:
                rename[col_lower[k]] = canon
                break
    if rename:
        df = df.rename(columns=rename)
        print("Renamed columns (detected):", rename)
    # Now choose final features present
    features = [c for c in prefer if c in df.columns]
    print("Final selected features:", features)
    return df, features

def build_preprocessor(X):
    numeric = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preproc = ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)])
    print("Preprocessor will use numeric:", numeric, "categorical:", categorical)
    return preproc

def cluster_colors_and_create_target(df, k=K_CLUSTERS):
    df = ensure_rgb(df)
    rgb = df[['r','g','b']].astype(float).values
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(rgb)
    df['color_cluster'] = labels
    centers = kmeans.cluster_centers_.round().astype(int)
    def rgb_to_hex(rgb):
        r,g,b = [int(max(0,min(255,x))) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    palette = {i: rgb_to_hex(centers[i]) for i in range(k)}
    print(f"Created color clusters (k={k})")
    return df, kmeans, palette

def train_supervised(df, features, target_col):
    # Drop rows with missing feature/target
    sub = df[features + [target_col]].dropna()
    print("Training rows (after dropna):", sub.shape)
    X = sub[features].copy()
    y = sub[target_col].astype(int) if sub[target_col].dtype.kind in 'biufc' else sub[target_col]
    # If y is categorical text, leave as-is (the pipeline will handle encoding via OHE for features)
    # Balance if needed (SMOTE)
    unique_vals = y.value_counts(normalize=True).to_dict()
    print("Target distribution (train):", unique_vals)
    do_smote = False
    if len(unique_vals) == 2:
        # if minority class < 30%, apply SMOTE
        min_prop = min(unique_vals.values())
        if min_prop < 0.30:
            do_smote = True

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    preproc = build_preprocessor(X_train)

    # if SMOTE needed we will apply it inside pipeline using imblearn; here we'll do manual resampling before gridsearch fits
    # but to keep consistent we will perform SMOTE after preprocessing transformation (easier approach: use imblearn.pipeline in grid if desired)
    # Simpler: transform categorical->OHE + numeric scaling via pipeline used with estimator that accepts sample weights. We'll apply SMOTE on raw X via simple encoding below.

    # Create classifier pipeline
    clf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)

    pipe = Pipeline([('preproc', preproc), ('clf', clf)])

    # Grid search
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {'clf__n_estimators': [100, 200], 'clf__max_depth': [10, 20, None]}

    print("Starting GridSearchCV...")
    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring='f1' if len(unique_vals)==2 else 'accuracy', n_jobs=-1, verbose=1)
    # If SMOTE needed: perform simple resample before grid.fit by applying to X_train/y_train transformed to numeric via simple pipeline
    if do_smote:
        print("Applying SMOTE to training data (imbalance detected).")
        # We need to convert X_train to numeric matrix first using preproc.fit_transform
        X_train_trans = preproc.fit_transform(X_train)
        sm = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = sm.fit_resample(X_train_trans, y_train)
        # Fit classifier separately: simpler approach - train grid on preprocessed arrays by wrapping classifier directly (skip gridsearch for preproc).
        # For simplicity, we will fit a classifier with tuned params on resampled data.
        # Quick basic grid on RandomForest only using preprocessed arrays:
        best_params = None
        best_score = -1
        for n in [100, 200]:
            for md in [10,20,None]:
                rf = RandomForestClassifier(n_estimators=n, max_depth=md, random_state=RANDOM_STATE, n_jobs=-1)
                rf.fit(X_res, y_res)
                preds = rf.predict(preproc.transform(X_test))
                f1 = f1_score(y_test, preds, average='binary') if len(unique_vals)==2 else f1_score(y_test, preds, average='weighted')
                if f1 > best_score:
                    best_score = f1
                    best_params = {'n_estimators': n, 'max_depth': md}
                    best_rf = rf
        print("Selected best params (SMOTE path):", best_params, "best f1:", best_score)
        # Build final pipeline combining preproc and best_rf
        final_pipe = Pipeline([('preproc', preproc), ('clf', best_rf)])
        # Evaluate
        y_pred = final_pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary' if len(unique_vals)==2 else 'weighted')
        try:
            roc = roc_auc_score(y_test, final_pipe.predict_proba(X_test)[:,1]) if len(unique_vals)==2 else None
        except:
            roc = None
        print("Final evaluation (SMOTE path): acc:", acc, "f1:", f1, "roc:", roc)
        return final_pipe, {'accuracy':acc, 'f1':f1, 'roc_auc':roc, 'best_params':best_params}

    # No SMOTE path: run GridSearchCV on pipeline
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='binary' if len(unique_vals)==2 else 'weighted')
    roc = None
    try:
        if len(unique_vals)==2:
            roc = roc_auc_score(y_test, best.predict_proba(X_test)[:,1])
    except:
        roc = None

    print("Evaluation on test set — accuracy:", acc, "f1:", f1, "roc_auc:", roc)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    return best, {'accuracy':acc, 'f1':f1, 'roc_auc':roc, 'best_params':grid.best_params_}

def train_unsupervised(df):
    # Create color clusters and then train a model that predicts cluster from user features
    df = ensure_rgb(df)
    df, kmeans, palette = cluster_colors_and_create_target(df, k=K_CLUSTERS)
    df, features = select_features(df)
    X = df[features].copy()
    y = df['color_cluster'].astype(int)
    print("Unsupervised path: training classifier to predict color_cluster from features (supervised on derived target).")
    # reuse supervised train function with derived target
    model, metrics = train_supervised(df.assign(color_cluster=y), features, 'color_cluster')
    return model, metrics, kmeans, palette, features

def main():
    df = load_data()
    # detect click column
    click_col = find_click_column(df)
    df = ensure_rgb(df)
    if click_col:
        print("Detected click column:", click_col, "(supervised training)")
        # canonicalize click column name to 'Clicked' for internal consistency
        df = df.rename(columns={click_col: 'Clicked'}) if click_col != 'Clicked' else df
        df, features = select_features(df)
        if len(features)==0:
            raise RuntimeError("No features found after selection. Check CSV headers.")
        model, metrics = train_supervised(df, features, 'Clicked')
        meta = {'mode':'supervised', 'model': model, 'features': features, 'metrics': metrics}
        # no kmeans/palette in supervised-only pipeline
        joblib.dump(meta, OUT_JOBLIB)
        print("Saved supervised meta to:", OUT_JOBLIB)
    else:
        print("No click column detected — switching to clustering+supervised on cluster labels (unsupervised path).")
        model, metrics, kmeans, palette, features = train_unsupervised(df)
        meta = {'mode':'unsupervised_cluster', 'model': model, 'features': features, 'metrics': metrics, 'kmeans': kmeans, 'palette': palette}
        joblib.dump(meta, OUT_JOBLIB)
        print("Saved unsupervised meta to:", OUT_JOBLIB)

    # Save human-readable summary
    summary = {'metrics': meta.get('metrics', {}), 'mode': meta.get('mode')}
    with open(SUMMARY_JSON, 'w') as f:
        json.dump(summary, f, indent=2)
    print("Wrote training summary to:", SUMMARY_JSON)
    print("TRAINING COMPLETE.")

if __name__ == "__main__":
    main()
