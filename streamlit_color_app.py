# streamlit_color_app.py
"""
Run:
streamlit run streamlit_color_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Color Recommender", layout="wide", page_icon="🎨")

MODEL_JOBLIB = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_recommender_cluster.joblib"
DATA_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"

# ---------- Load model & data ----------
if not os.path.exists(MODEL_JOBLIB):
    st.error(f"Model not found at {MODEL_JOBLIB}. First run train_color_recommender.py")
    st.stop()

meta = joblib.load(MODEL_JOBLIB)
model = meta['model']
features = meta['features']
kmeans = meta['kmeans']
palette = meta['cluster_palette']
cluster_examples = meta.get('cluster_examples', {})

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not load dataset at {DATA_PATH}: {e}")
    st.stop()

# Clean/ensure RGB columns
if not set(['r','g','b']).issubset(df.columns) and 'hex_code' in df.columns:
    def hex_to_rgb(h):
        try:
            h = str(h).lstrip('#')
            return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        except:
            return 128,128,128
    df[['r','g','b']] = pd.DataFrame(df['hex_code'].apply(hex_to_rgb).tolist(), index=df.index)

# Header
st.markdown("<h1 style='text-align:center; color:#EF476F;'>🎨 Smart Color Recommendation Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6c757d;'>Interactive DAV dashboard — analysis, predictions and color previews</p>", unsafe_allow_html=True)
st.write("")

# Sidebar inputs
st.sidebar.header("Input user attributes")
sidebar_inputs = {}
# use features list to create widgets
for feat in features:
    if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
        mini, maxi = int(df[feat].min()), int(df[feat].max())
        median = int(df[feat].median())
        sidebar_inputs[feat] = st.sidebar.slider(feat, mini, maxi, median)
    elif feat in df.columns:
        opts = sorted(df[feat].dropna().unique().tolist())
        if len(opts) > 200:
            opts = opts[:200]
        sidebar_inputs[feat] = st.sidebar.selectbox(feat, opts)
    else:
        # fallback text input
        sidebar_inputs[feat] = st.sidebar.text_input(feat, value="unknown")

# additional RGB preview sliders (redundant if r/g/b are in features)
r = st.sidebar.slider("R", 0, 255, 120)
g = st.sidebar.slider("G", 0, 255, 120)
b = st.sidebar.slider("B", 0, 255, 120)
sidebar_inputs['r'] = r; sidebar_inputs['g'] = g; sidebar_inputs['b'] = b

st.sidebar.markdown("---")
st.sidebar.write("Model: RandomForest + color clustering")
st.sidebar.write(f"Color families: {len(palette)}")

# Build input_df
input_df = pd.DataFrame([sidebar_inputs], columns=features)

# Main layout: left input & prediction, right visuals
left, right = st.columns([1,2])

with left:
    st.subheader("Input preview")
    st.table(input_df.T.rename(columns={0:"value"}))
    st.markdown("**Selected color preview**")
    hex_preview = "#{:02x}{:02x}{:02x}".format(r,g,b)
    st.markdown(f"<div style='width:160px;height:80px;border-radius:8px;background:{hex_preview};border:1px solid #ddd'></div>", unsafe_allow_html=True)
    st.write("Hex:", hex_preview)

    if st.button("Recommend color"):
        try:
            probs = model.predict_proba(input_df)[0]
            top = int(np.argmax(probs))
            prob_top = float(probs[top])
            rep_hex = palette.get(top, "#777777")
            st.success(f"Recommended color family: **Cluster {top}** — probability {prob_top:.2f}")
            st.markdown(f"Representative color: <div style='display:inline-block;margin-left:10px'>{'<div style=\"width:70px;height:40px;border-radius:6px;background:'+rep_hex+';border:1px solid #ccc\"></div>'}</div>", unsafe_allow_html=True)
            st.write("Representative hex:", rep_hex)
            ex = cluster_examples.get(top, [])
            if ex:
                st.write("Example color names:", ", ".join(ex))
            # show top 3
            top3 = np.argsort(probs)[-3:][::-1]
            st.write("Top clusters & probabilities:", {int(i):float(probs[int(i)]) for i in top3})
        except Exception as e:
            st.error("Prediction failed: " + str(e))

with right:
    st.subheader("Dataset analytics")
    st.markdown("### Click distribution")
    # try to detect click column
    click_col = None
    for c in df.columns:
        if 'click' in c.lower():
            click_col = c
            break
    if click_col is None:
        st.warning("No click column found in dataset; visualizations will use counts.")
        temp = df.copy()
        temp['__one'] = 1
        count = temp.groupby('Product_Category').size().reset_index(name='count')
        fig = px.bar(count, x='Product_Category', y='count', title="Rows per Product Category")
        st.plotly_chart(fig, use_container_width=True)
    else:
        click_counts = df[click_col].value_counts().reset_index()
        click_counts.columns = ['Click','Count']
        fig = px.bar(click_counts, x='Click', y='Count', color='Click', title="Click Distribution")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top colors (by frequency)")
    if 'color_name' in df.columns:
        top_colors = df['color_name'].value_counts().head(15).reset_index()
        top_colors.columns = ['Color','Count']
        fig = px.bar(top_colors, x='Color', y='Count', title="Top 15 Color Names")
        st.plotly_chart(fig, use_container_width=True)
    elif 'Interface_Color' in df.columns:
        top_colors = df['Interface_Color'].value_counts().head(15).reset_index()
        top_colors.columns = ['Color','Count']
        fig = px.bar(top_colors, x='Color', y='Count', title="Top 15 Interface Colors")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No color_name or Interface_Color column available.")

    st.markdown("---")
    st.markdown("### RGB scatter (R vs G colored by cluster)")
    if set(['r','g']).issubset(df.columns):
        pts = df[['r','g']].copy()
        pts['cluster'] = kmeans.predict(df[['r','g','b']].astype(float).values)
        pts['hex'] = pts['cluster'].map(palette)
        fig = px.scatter(pts, x='r', y='g', color=pts['cluster'].astype(str), color_discrete_sequence=[palette[i] for i in sorted(palette.keys())], title="R vs G colored by cluster")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write("No RGB columns for scatter plot.")

    st.markdown("---")
    st.markdown("### Correlation heatmap (numeric features)")
    numcols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if len(numcols) >= 2:
        corr = df[numcols].corr()
        fig2 = plt.figure(figsize=(6,4))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='RdYlBu')
        st.pyplot(fig2)
    else:
        st.write("Not enough numeric columns for heatmap.")

# Footer
st.markdown("---")
st.markdown("<div style='text-align:center;color:gray;'>Made with ❤️ by Priyaa — DAV Project</div>", unsafe_allow_html=True)
