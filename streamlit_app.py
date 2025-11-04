# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import io
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from pathlib import Path
import colorsys
import traceback

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "color_click_recommender_rf.joblib")
DATA_PATH = os.path.join(BASE_DIR, "augmented_color_click_dataset.csv")
TOP_CANDIDATES = 20
# ----------------------------------------

st.set_page_config(page_title="Color Clicks — Smart Recommender",
                   layout="wide")

# ---- CSS / Theme ----
st.markdown(
    """
    <style>
      /* Pastel-light professional look */
      .title {font-size:56px; color:#ffffff; font-weight:800; text-align:center; padding-bottom:6px;}
      .subtitle {font-size:32px; color:#666; text-align:center; margin-bottom:18px;}
      .card {background:linear-gradient(180deg,#ffffff,#fbfbff); padding:14px; border-radius:12px; box-shadow:0 6px 30px rgba(40,40,90,0.04);}
      .metric {background:#fff; border-radius:8px; padding:8px;}
      /* Recommend button hover animation */
      .recommend-btn button {
        background: linear-gradient(90deg,#ff8fa3,#ffb26b);
        color: white;
        border-radius: 10px;
        padding: 8px 18px;
        font-weight:700;
      }
      .recommend-btn button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(255,130,120,0.18);
      }
      /* Small nice card for inputs */
      .input-card {padding:10px; border-radius:10px; background: #ffffff; box-shadow:0 6px 18px rgba(0,0,0,0.03);}
      /* Tabs alignment */
      .stTabs [role="tablist"] {gap: 6px;}
    </style>
    """, unsafe_allow_html=True
)

# ---- Header ----
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("<div class='title'>Color Clicks — Smart Recommendation Dashboard</div>", unsafe_allow_html=True)

with col_h2:
    pass

# ---------- Load model & data ----------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}\nRun training first and ensure path is correct.")
    st.stop()

meta = joblib.load(MODEL_PATH)
mode = meta.get('mode', 'supervised' if 'model' in meta else 'unsupervised_cluster')
model = meta.get('model')
features = meta.get('features', []) or []
kmeans = meta.get('kmeans', None)
palette = meta.get('palette', {})  # cluster_idx -> hex
metrics = meta.get('metrics', {})

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset at: {DATA_PATH}\n{e}")
    st.stop()

# ---------- Ensure r,g,b exist ----------
def ensure_rgb_from_hex(df):
    if set(['r','g','b']).issubset(df.columns):
        return df
    hex_col = next((c for c in df.columns if 'hex' in c.lower()), None)
    if hex_col:
        def hex_to_rgb(h):
            try:
                s=str(h).lstrip('#')
                return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
            except:
                return (128,128,128)
        df[['r','g','b']] = pd.DataFrame(df[hex_col].apply(hex_to_rgb).tolist(), index=df.index)
    else:
        df['r'], df['g'], df['b'] = 128,128,128
    return df

df = ensure_rgb_from_hex(df)

# normalize time column if variants exist
time_col_candidates = ['Time_Spent_sec','time_spent','Time_Spent','time_spent_sec']
time_col = next((c for c in time_col_candidates if c in df.columns), None)
if time_col and time_col != 'Time_Spent_sec':
    df['Time_Spent_sec'] = df[time_col]

# click column detection
click_col = next((c for c in df.columns if 'click' in c.lower() or c.lower() in ('clicked','is_clicked')), None)

# convert common column names to lower-case keys to check presence
cols_lower = {c.lower(): c for c in df.columns}

# ---------------- Sidebar Inputs ----------------
st.sidebar.header("User & Context Inputs")
def opts(col, fallback):
    return sorted(df[col].dropna().unique().tolist()) if col in df.columns else fallback

age_min = int(df['age'].min()) if 'age' in df.columns else 10
age_max = int(df['age'].max()) if 'age' in df.columns else 70
age_default = int(df['age'].median()) if 'age' in df.columns else 25
age = st.sidebar.slider("Age", age_min, age_max, age_default)

gender = st.sidebar.selectbox("Gender", opts('gender', ['Female','Male','Other']))
device = st.sidebar.selectbox("Device Type", opts('device_type', ['Mobile','Desktop','Tablet']))
product = st.sidebar.selectbox("Product Category", opts('Product_Category', ['Fashion','Tech','Home','Food','Sports']))
mood = st.sidebar.selectbox("User Mood", opts('Mood', ['Happy','Sad','Calm','Neutral']))
season = st.sidebar.selectbox("Season", opts('Season', ['Summer','Winter','Spring','Autumn','Monsoon']))

time_min = int(df['Time_Spent_sec'].min()) if 'Time_Spent_sec' in df.columns else 1
time_max = int(df['Time_Spent_sec'].max()) if 'Time_Spent_sec' in df.columns else 300
time_spent = st.sidebar.slider("Time Spent (sec)", time_min, time_max, int((time_min+time_max)//10))

st.sidebar.markdown("---")
r = st.sidebar.slider("R (preview)", 0, 255, 200)
g = st.sidebar.slider("G (preview)", 0, 255, 120)
b = st.sidebar.slider("B (preview)", 0, 255, 80)
st.sidebar.markdown("---")

st.sidebar.write("Model mode:", mode)
if metrics:
    st.sidebar.write("Model metrics (sample):", metrics)

st.sidebar.info("Why RGB? RGB numeric values precisely encode colors for the model. Why Time Spent? Time spent captures engagement and helps predict clicks together with color.")

# build user_row dictionary
user_row = {}
for f in features:
    lf = f.lower()
    if 'age' in lf:
        user_row[f] = age
    elif 'gender' in lf:
        user_row[f] = gender
    elif 'device' in lf:
        user_row[f] = device
    elif 'product' in lf or 'category' in lf:
        user_row[f] = product
    elif 'mood' in lf:
        user_row[f] = mood
    elif 'season' in lf:
        user_row[f] = season
    elif 'time' in lf:
        user_row[f] = time_spent
    elif f in ['r','g','b']:
        user_row[f] = 0
    else:
        user_row[f] = df[f].mode().iloc[0] if f in df.columns else 0

input_preview_df = pd.DataFrame([user_row])

# ---------------- Tabs Layout ----------------
tabs = st.tabs(["🏠 Overview", "📈 Color Analytics", "👥 User Behavior", "⏱ Engagement Trends", "💡 Insights"])

# ---------- Helper functions ----------
def hex_to_rgb_tuple(h):
    s=str(h).lstrip('#')
    try:
        return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    except:
        return (128,128,128)

def safe_predict_proba(X):
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:,1]
            else:
                return np.max(proba, axis=1)
        except:
            pass
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return np.array(preds, dtype=float)
    return np.zeros(len(X))

# ---------------- Tab 0: Overview ----------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Input summary")
    st.table(input_preview_df.T.rename(columns={0:"value"}))
    st.markdown("**Selected color preview (preview sliders)**")
    hex_preview = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
    st.markdown(
        f"<div style='width:260px;height:120px;border-radius:12px;background:{hex_preview};box-shadow:0 6px 18px rgba(0,0,0,0.06);border:1px solid rgba(0,0,0,0.06)'></div>",
        unsafe_allow_html=True
    )
    st.write("Hex:", hex_preview)
    st.markdown("---")

    st.markdown("<div class='recommend-btn'>", unsafe_allow_html=True)
    if st.button("🔮 Recommend best color"):
        st.info("Generating recommendations…")
        try:
            if mode == 'supervised':
                candidates = []
                if isinstance(palette, dict) and len(palette)>0:
                    candidates.extend(list(palette.values()))
                if 'hex_code' in df.columns:
                    candidates += df['hex_code'].dropna().astype(str).unique().tolist()
                if len(candidates) < 5:
                    centers = KMeans(n_clusters=min(12, max(3, len(df))), random_state=1, n_init=10).fit(df[['r','g','b']].values).cluster_centers_.round().astype(int)
                    for c in centers:
                        candidates.append(f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}")
                uniq=[]
                for h in candidates:
                    if h not in uniq:
                        uniq.append(h)
                candidates = uniq[:max(TOP_CANDIDATES, 12)]

                rows=[]
                for hexc in candidates:
                    rr,gg,bb = hex_to_rgb_tuple(hexc)
                    row = user_row.copy()
                    if 'r' in features: row['r']=rr
                    if 'g' in features: row['g']=gg
                    if 'b' in features: row['b']=bb
                    Xc = pd.DataFrame([row])
                    for col in features:
                        if col not in Xc.columns:
                            Xc[col] = df[col].mode().iloc[0] if col in df.columns else 0
                    Xc = Xc[features]
                    try:
                        p_arr = safe_predict_proba(Xc)
                        p_click = float(p_arr[0]) if len(p_arr)>0 else ((rr+gg+bb)/3)/255
                    except:
                        p_click = ((rr+gg+bb)/3)/255
                    rows.append((hexc, rr, gg, bb, p_click))
                rows_sorted = sorted(rows, key=lambda x: x[4], reverse=True)
                best_hex, br, bg, bb_, best_prob = rows_sorted[0]
                st.success(f"Recommended color: **{best_hex.upper()}** — predicted click score **{best_prob:.2f}**")
                st.markdown(f"<div style='width:240px;height:120px;border-radius:12px;background:{best_hex};box-shadow:0 6px 18px rgba(0,0,0,0.06);border:1px solid rgba(0,0,0,0.06)'></div>", unsafe_allow_html=True)
                st.write("Top candidate colors:")
                cols = st.columns(5)
                for i,(h,rr,gg,bb,p) in enumerate(rows_sorted[:5]):
                    with cols[i%5]:
                        st.markdown(f"<div style='width:80px;height:48px;border-radius:8px;background:{h};border:1px solid #ddd'></div>", unsafe_allow_html=True)
                        st.caption(f"{h} — {p:.2f}")
            else:
                row = user_row.copy()
                if 'r' in features: row['r']=r
                if 'g' in features: row['g']=g
                if 'b' in features: row['b']=b
                Xc = pd.DataFrame([row])
                for col in features:
                    if col not in Xc.columns:
                        Xc[col] = df[col].mode().iloc[0] if col in df.columns else 0
                Xc = Xc[features]
                try:
                    pred = model.predict(Xc)[0]
                    hex_rep = palette.get(int(pred), "#777777")
                    st.success(f"Recommended color family: Cluster {int(pred)} — representative {hex_rep}")
                    st.markdown(f"<div style='width:240px;height:120px;border-radius:12px;background:{hex_rep};box-shadow:0 6px 18px rgba(0,0,0,0.06);border:1px solid rgba(0,0,0,0.06)'></div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error("Prediction failed: " + str(e))
                    st.write(traceback.format_exc())
        except Exception as e:
            st.error("Recommendation pipeline failed: " + str(e))
            st.write(traceback.format_exc())
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Tab 1: Color Analytics ----------------
with tabs[1]:
    st.subheader("Color Analytics — RGB & Clusters")
    st.markdown("### RGB distributions")
    fig_r = px.histogram(df, x='r', nbins=30, title="Distribution of R channel")
    fig_g = px.histogram(df, x='g', nbins=30, title="Distribution of G channel")
    fig_b = px.histogram(df, x='b', nbins=30, title="Distribution of B channel")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(fig_r, use_container_width=True)
    c2.plotly_chart(fig_g, use_container_width=True)
    c3.plotly_chart(fig_b, use_container_width=True)
