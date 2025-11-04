# train_model.py
"""
Train workflow:
1) Load dataset (detect column names flexibly)
2) Ensure r,g,b exist (parse hex_code if needed)
3) KMeans cluster colors into K families
4) Train RandomForest classifier to predict cluster from user/context features
5) Save meta dict with model, features, kmeans, palette, encoders -> joblib
Run:
    python train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

# -------- CONFIG --------
DATA_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"
OUT_JOBLIB = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_recommender_cluster.joblib"
K_CLUSTERS = 12
RANDOM_STATE = 42
# ------------------------

def load_df(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    print("Columns:", df.columns.tolist())
    return df

def ensure_rgb(df):
    # If r,g,b exist, keep. Else try hex_code or hex, else fill 128.
    if set(['r','g','b']).issubset(df.columns):
        return df
    hex_col = None
    for c in df.columns:
        if 'hex' in c.lower():
            hex_col = c
            break
    if hex_col:
        def hex_to_rgb(h):
            try:
                s = str(h).lstrip('#')
                return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
            except:
                return (128,128,128)
        rgb = df[hex_col].apply(hex_to_rgb).tolist()
        df[['r','g','b']] = pd.DataFrame(rgb, index=df.index)
    else:
        # fallback: neutral mid-gray
        df['r'], df['g'], df['b'] = 128,128,128
    return df

def make_clusters(df, k=K_CLUSTERS):
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
    return df, kmeans, palette

def select_features(df):
    # Preferred columns list - adjust if your CSV uses slightly different names
    candidates = ['age','Age','gender','Gender','device_type','Device_Type','ad_position','adPosition',
                  'Product_Category','ProductCategory','color_name','Mood','Season','Time_Spent_sec','Time_Spent']
    # Normalize: map actual present names to canonical names used in training
    col_map = {}
    # We'll check the dataset and pick standard names
    # Build a small mapping
    rename = {}
    if 'age' not in df.columns and 'Age' in df.columns:
        rename['Age'] = 'age'
    if 'gender' not in df.columns and 'Gender' in df.columns:
        rename['Gender'] = 'gender'
    if 'device_type' not in df.columns:
        if 'Device_Type' in df.columns:
            rename['Device_Type'] = 'device_type'
        elif 'device' in df.columns:
            rename['device'] = 'device_type'
    if 'Product_Category' not in df.columns and 'ProductCategory' in df.columns:
        rename['ProductCategory'] = 'Product_Category'
    if 'Time_Spent_sec' not in df.columns and 'Time_Spent' in df.columns:
        rename['Time_Spent'] = 'Time_Spent_sec'
    if rename:
        df = df.rename(columns=rename)
    # Final feature list - choose only those present
    preferred = ['age','gender','device_type','ad_position','Product_Category','Mood','Season','r','g','b','Time_Spent_sec']
    features = [c for c in preferred if c in df.columns]
    print("Selected features:", features)
    return df, features

def build_preprocessor(X):
    numeric = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preproc = ColumnTransformer([('num', num_pipe, numeric), ('cat', cat_pipe, categorical)])
    print("Numeric:", numeric, "Categorical:", categorical)
    return preproc

def train():
    df = load_df()
    df, kmeans, palette = make_clusters(df, K_CLUSTERS)
    df, features = select_features(df)
    if len(features) == 0:
        raise ValueError("No recognized feature columns found in dataset. Check headers.")
    X = df[features].copy()
    y = df['color_cluster'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    preproc = build_preprocessor(X_train)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([('preproc', preproc), ('clf', clf)])

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {'clf__max_depth': [10, 20, None]}
    grid = GridSearchCV(pipe, param_grid, cv=skf, n_jobs=-1, verbose=1)
    print("Starting GridSearch (may take few minutes)...")
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred))

    # Save meta object for Streamlit
    meta = {
        'model': best,
        'features': features,
        'kmeans': kmeans,
        'palette': palette
    }
    joblib.dump(meta, OUT_JOBLIB)
    print("Saved model and meta to:", OUT_JOBLIB)
    return OUT_JOBLIB

if __name__ == "__main__":
    train()
