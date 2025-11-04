
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
from sklearn.metrics import classification_report
import joblib

# ---------------- CONFIG ----------------
DATA_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"
OUT_JOBLIB = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_recommender_cluster.joblib"
K_CLUSTERS = 12           # how many color families to create (changeable)
RANDOM_STATE = 42
# ----------------------------------------

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    print("Loaded:", df.shape)
    return df

def ensure_rgb(df):
    # ensure r,g,b exist; if not, try to parse hex_code
    if not set(['r','g','b']).issubset(df.columns):
        if 'hex_code' in df.columns:
            def hex_to_rgb(h):
                try:
                    h = str(h).lstrip('#')
                    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
                except:
                    return 0,0,0
            rgb = df['hex_code'].apply(hex_to_rgb).tolist()
            df[['r','g','b']] = pd.DataFrame(rgb, index=df.index)
        else:
            # fallback: set neutral gray
            df['r'], df['g'], df['b'] = 128,128,128
    return df

def make_color_clusters(df, k=K_CLUSTERS):
    df = ensure_rgb(df)
    rgb = df[['r','g','b']].astype(float).values
    print("Clustering RGB of shape:", rgb.shape)
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(rgb)
    df['color_cluster'] = labels
    # cluster palette: hex of centers
    centers = kmeans.cluster_centers_.round().astype(int)
    def rgb_to_hex(rgb):
        r,g,b = [int(max(0,min(255,x))) for x in rgb]
        return f"#{r:02x}{g:02x}{b:02x}"
    cluster_palette = {i: rgb_to_hex(centers[i]) for i in range(k)}
    return df, kmeans, cluster_palette

def select_features(df):
    # prefer these columns if present
    candidate = ['age','gender','device_type','ad_position','Product_Category','Mood','Season','r','g','b','Time_Spent_sec']
    features = [c for c in candidate if c in df.columns]
    print("Using features:", features)
    return features

def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object','category','bool']).columns.tolist()

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    # use sparse_output for modern sklearn compatibility
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preprocessor = ColumnTransformer([('num', num_pipe, numeric_features),
                                      ('cat', cat_pipe, categorical_features)])
    print("Numeric:", numeric_features, "Categorical:", categorical_features)
    return preprocessor

def train_and_save():
    df = load_data()
    df, kmeans, palette = make_color_clusters(df, k=K_CLUSTERS)

    features = select_features(df)
    if len(features) == 0:
        raise ValueError("No feature columns found. Check dataset headers.")

    X = df[features].copy()
    y = df['color_cluster'].astype(int)

    # Save some example names per cluster (for display)
    cluster_examples = {}
    if 'color_name' in df.columns:
        for c in range(K_CLUSTERS):
            names = df.loc[df['color_cluster']==c, 'color_name'].dropna().unique()[:5].tolist()
            cluster_examples[c] = names
    else:
        for c in range(K_CLUSTERS):
            cluster_examples[c] = []

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    preprocessor = build_preprocessor(X_train)

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)

    pipe = Pipeline([('preproc', preprocessor), ('clf', clf)])

    # small grid search (keeps short)
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    param_grid = {'clf__max_depth': [10, 20, None]}
    grid = GridSearchCV(pipe, param_grid, cv=skf, n_jobs=-1, verbose=1)
    print("Fitting model (GridSearchCV)... this may take a few minutes")
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    y_pred = best.predict(X_test)
    print("Classification report:\n", classification_report(y_test, y_pred))

    meta = {
        'model': best,
        'features': features,
        'kmeans': kmeans,
        'cluster_palette': palette,
        'cluster_examples': cluster_examples
    }
    joblib.dump(meta, OUT_JOBLIB)
    print("Saved meta to:", OUT_JOBLIB)
    return OUT_JOBLIB

if __name__ == "__main__":
    train_and_save()