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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "color_click_recommender_rf.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "augmented_color_click_dataset.csv")
TOP_CANDIDATES = 20
# ----------------------------------------

st.set_page_config(page_title="Color Clicks — Smart Recommender",
                   layout="wide")

# ---- CSS / Theme ----
st.markdown(
    """
    <style>
      /* Pastel-light professional look */
      .title {font-size:56px; color:#
; font-weight:800; text-align:
Center; padding-bottom:6px;}
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
      /* Tabs alignment (approx) - placing tabs on right column */
      .stTabs [role="tablist"] {gap: 6px;}
    </style>
    """, unsafe_allow_html=True
)

# ---- Header ----
col_h1, col_h2 = st.columns([3,1])
with col_h1:
    st.markdown("<div class='title'>Color Clicks — Smart Recommendation Dashboard</div>", unsafe_allow_html=True)

with col_h2:
    # small status box (model & rows)
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

# build user_row dictionary for model features (fill missing features with dataset mode / sensible defaults)
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
        # Align with training column name — commonly Time_Spent_sec or similar
        if 'Time_Spent_sec' in df.columns and 'Time_Spent_sec' in features:
            user_row[f] = time_spent
        else:
            user_row[f] = time_spent
    elif f in ['r','g','b']:
        user_row[f] = 0
    else:
        user_row[f] = df[f].mode().iloc[0] if f in df.columns else 0

input_preview_df = pd.DataFrame([user_row])

# ---------------- Tabs Layout (placed in right column area visually) ----------------
# Create a layout so tabs appear in a wide central area
tabs = st.tabs(["🏠 Overview", "📈 Color Analytics", "👥 User Behavior", "⏱ Engagement Trends", "💡 Insights"])

# ---------- Helper functions ----------
def hex_to_rgb_tuple(h):
    s=str(h).lstrip('#')
    try:
        return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)
    except:
        return (128,128,128)

def safe_predict_proba(X):
    """Try predict_proba, if not available fallback to predict or heuristic scores."""
    # if model has predict_proba
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            # if binary, return prob of positive class for each row
            if proba.shape[1] == 2:
                return proba[:,1]
            else:
                # multi-class: take max probability as proxy
                return np.max(proba, axis=1)
        except Exception as e:
            raise
    # else fallback to predict (0/1)
    if hasattr(model, "predict"):
        preds = model.predict(X)
        return np.array(preds, dtype=float)
    # ultimate fallback - return zeros
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

    # Recommend button block with animation CSS
    st.markdown("<div class='recommend-btn'>", unsafe_allow_html=True)
    if st.button("🔮 Recommend best color"):
        st.info("Generating recommendations…")
        try:
            if mode == 'supervised':
                # build candidate list
                candidates = []
                if isinstance(palette, dict) and len(palette)>0:
                    candidates.extend(list(palette.values()))
                if 'hex_code' in df.columns:
                    candidates += df['hex_code'].dropna().astype(str).unique().tolist()
                if len(candidates) < 5:
                    centers = KMeans(n_clusters=min(12, max(3, len(df))), random_state=1, n_init=10).fit(df[['r','g','b']].values).cluster_centers_.round().astype(int)
                    for c in centers:
                        candidates.append(f"#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}")
                # dedupe
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
                    # create DataFrame with exact columns and order
                    Xc = pd.DataFrame([row])
                    # ensure all features present (fill missing with dataset mode or 0)
                    for col in features:
                        if col not in Xc.columns:
                            Xc[col] = df[col].mode().iloc[0] if col in df.columns else 0
                    Xc = Xc[features]
                    try:
                        p_arr = safe_predict_proba(Xc)
                        p_click = float(p_arr[0]) if len(p_arr)>0 else ((rr+gg+bb)/3)/255
                        # normalize if predict returned 0/1 -> keep as is
                    except Exception as e:
                        # fallback heuristic (brightness)
                        p_click = ((rr+gg+bb)/3)/255
                    rows.append((hexc, rr, gg, bb, p_click))
                # choose best
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
                # unsupervised path: predict cluster from preview rgb
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

    st.markdown("---")
    st.markdown("### Color clusters & click rate")
    df_vis = df.copy()
    if kmeans is None:
        k = min(12, max(3, len(df)//500))
        k = max(3, k)
        kmeans_vis = KMeans(n_clusters=k, random_state=1, n_init=10)
        df_vis['cluster'] = kmeans_vis.fit_predict(df_vis[['r','g','b']].values)
        centers_vis = kmeans_vis.cluster_centers_.round().astype(int)
        palette_vis = {i:f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for i,c in enumerate(centers_vis)}
    else:
        df_vis['cluster'] = kmeans.predict(df_vis[['r','g','b']].values)
        palette_vis = palette

    if click_col:
        cluster_stats = df_vis.groupby('cluster')[click_col].mean().reset_index().rename(columns={click_col:'click_rate'})
        cluster_stats['hex'] = cluster_stats['cluster'].map(palette_vis)
        fig_cluster = px.bar(cluster_stats, x='cluster', y='click_rate', color='hex',
                             color_discrete_map={h:h for h in cluster_stats['hex']}, title="Cluster vs Click Rate")
        st.plotly_chart(fig_cluster, use_container_width=True)
    else:
        cluster_counts = df_vis['cluster'].value_counts().reset_index()
        cluster_counts.columns = ['cluster','count']
        fig_cluster = px.bar(cluster_counts, x='cluster', y='count', title="Cluster counts")
        st.plotly_chart(fig_cluster, use_container_width=True)

    st.markdown("---")
    st.markdown("### 3D RGB view (sampled)")
    sample = df_vis.sample(min(2000, len(df_vis)), random_state=1)
    fig_3d = px.scatter_3d(sample, x='r', y='g', z='b',
                          color=click_col if click_col else 'cluster',
                          hover_data=['r','g','b'],
                          title="3D RGB scatter (sampled)")
    st.plotly_chart(fig_3d, use_container_width=True)

# ---------------- Tab 2: User Behavior ----------------
with tabs[2]:
    st.subheader("User Behavior — Demographics & Preferences")
    # Age groups
    if 'age' in df.columns and click_col:
        st.markdown("### Click rate by age group")
        df['age_group'] = pd.cut(df['age'], bins=[0,18,25,35,50,100], labels=['<18','18-25','25-35','35-50','50+'])
        ag = df.groupby('age_group')[click_col].mean().reset_index()
        st.plotly_chart(px.bar(ag, x='age_group', y=click_col, title="Click rate by age group"), use_container_width=True)
    # Gender
    if 'gender' in df.columns and click_col:
        st.markdown("### Click rate by gender")
        gg = df.groupby('gender')[click_col].mean().reset_index()
        st.plotly_chart(px.bar(gg, x='gender', y=click_col, title="Click rate by gender"), use_container_width=True)
    # Mood heatmap
    # Accept both 'Mood' and 'mood' columns (case sensitivity)
    mood_col = next((c for c in df.columns if c.lower() == 'mood'), None)
    if mood_col:
        st.markdown("### Mood vs average RGB (heatmap)")
        heat = df.groupby(mood_col)[['r','g','b']].mean()
        fig_heat = px.imshow(heat, labels=dict(x='RGB', y='Mood', color='mean'), x=['r','g','b'], y=heat.index, title="Mood vs avg RGB")
        st.plotly_chart(fig_heat, use_container_width=True)
    # Product
    if 'Product_Category' in df.columns and click_col:
        st.markdown("### Click rate by product category")
        pc = df.groupby('Product_Category')[click_col].mean().reset_index().sort_values(click_col, ascending=False)
        st.plotly_chart(px.pie(pc, values=click_col, names='Product_Category', title="Click rate per product category"), use_container_width=True)

# ---------------- Tab 3: Engagement Trends ----------------
with tabs[3]:
    st.subheader("Engagement Trends — Time & Brightness")

    # Time spent vs click: use string labels for bins to avoid Interval JSON problem
    if 'Time_Spent_sec' in df.columns and click_col:
        st.markdown("### Time Spent vs Click (binned)")
        df_time = df.copy()
        df_time['time_bin'] = pd.cut(df_time['Time_Spent_sec'], bins=10)
        time_agg = df_time.groupby('time_bin')[click_col].mean().reset_index()
        # Convert Interval to strings for plotly
        time_agg['time_bin_str'] = time_agg['time_bin'].astype(str)
        fig_time = px.line(time_agg, x='time_bin_str', y=click_col, markers=True, title="Click rate across time spent bins")
        fig_time.update_xaxes(title="Time spent bin")
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("Add `Time_Spent_sec` column to enable Time vs Click visualizations.")

    st.markdown("---")
    st.markdown("### Brightness vs Click Rate")
    df['brightness'] = df[['r','g','b']].mean(axis=1)
    if click_col:
        bright_agg = df.groupby(pd.cut(df['brightness'], bins=12))[click_col].mean().reset_index()
        bright_agg['brightness_str'] = bright_agg['brightness'].astype(str)
        fig_b = px.bar(bright_agg, x='brightness_str', y=click_col, title="Brightness bucket vs Click Rate")
        fig_b.update_xaxes(title="Brightness bin")
        st.plotly_chart(fig_b, use_container_width=True)
    else:
        st.plotly_chart(px.histogram(df, x='brightness', nbins=20, title="Brightness distribution"), use_container_width=True)

    st.markdown("---")
    st.markdown("### Correlation heatmap (numeric features)")
    numeric_cols = ['r','g','b']
    if 'Time_Spent_sec' in df.columns:
        numeric_cols.append('Time_Spent_sec')
    if click_col:
        numeric_cols.append(click_col)
    corr = df[numeric_cols].corr()
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu', title="Correlation matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------- Tab 4: Insights ----------------
with tabs[4]:
    st.subheader("Auto-generated Insights")
    insights = []
    try:
        if click_col and 'cluster' in df_vis.columns:
            cs = df_vis.groupby('cluster')[click_col].mean().reset_index()
            top_cluster = int(cs.sort_values(click_col, ascending=False).iloc[0]['cluster'])
            insights.append(f"Cluster {top_cluster} shows the highest average click probability.")
    except Exception:
        pass
    if 'Mood' in df.columns:
        try:
            mood_avg = df.groupby('Mood')[['r','g','b']].mean()
            mood_avg['brightness'] = mood_avg.mean(axis=1)
            max_mood = mood_avg['brightness'].idxmax()
            insights.append(f"Users labeled '{max_mood}' prefer brighter colors on average.")
        except Exception:
            pass
    if 'Product_Category' in df.columns and click_col:
        try:
            best_prod = df.groupby('Product_Category')[click_col].mean().idxmax()
            insights.append(f"Products in '{best_prod}' category attract the most clicks on average.")
        except Exception:
            pass

    if not insights:
        st.write("No strong insight derived from current dataset.")
    else:
        for it in insights:
            st.write("•", it)

# ---------------- Footer / end ----------------
st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
