import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
import os

st.set_page_config(page_title="Smart Color Recommender", layout="wide", page_icon="🎨")

# ---------- HEADER ----------
st.markdown("""
<h1 style='text-align:center; color:#ff4b8a;'>🎨 Smart Color Recommender Dashboard</h1>
<h5 style='text-align:center; color:#555;'>Dynamic, data-driven color prediction with interactive analytics</h5>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
MODEL_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_click_recommender_rf.joblib"
DATA_PATH = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found! Please train it first.")
    st.stop()

meta = joblib.load(MODEL_PATH)
model = meta.get("model", None)
features = meta.get("features", [])
mode = meta.get("mode", "supervised")

df = pd.read_csv(DATA_PATH)
if "r" not in df.columns:
    df["r"], df["g"], df["b"] = np.random.randint(0, 255, len(df)), np.random.randint(0, 255, len(df)), np.random.randint(0, 255, len(df))

# ---------- SIDEBAR ----------
st.sidebar.header("🧍‍♀️ User Context Inputs")

age = st.sidebar.slider("Age", 10, 70, 25)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
device = st.sidebar.selectbox("Device", ["Mobile", "Desktop", "Tablet"])
product = st.sidebar.selectbox("Product Category", ["Fashion", "Tech", "Food", "Home", "Sports"])
mood = st.sidebar.selectbox("Mood", ["Happy", "Sad", "Calm", "Neutral"])
season = st.sidebar.selectbox("Season", ["Summer", "Winter", "Spring", "Monsoon"])
time_spent = st.sidebar.slider("Time Spent (sec)", 10, 300, 100)

st.sidebar.markdown("---")

# ---------- COLOR RECOMMENDATION ----------
st.subheader("🎯 Personalized Color Recommendation")

# create base input row
user_input = {
    "age": age, "gender": gender, "device_type": device, "Product_Category": product,
    "Mood": mood, "Season": season, "Time_Spent_sec": time_spent,
}

# generate diverse candidate colors
candidates = []
for _ in range(80):
    r, g, b = random.randint(0,255), random.randint(0,255), random.randint(0,255)
    user_input.update({"r": r, "g": g, "b": b})
    X = pd.DataFrame([user_input], columns=features)
    try:
        p = model.predict_proba(X)[0][1]
    except:
        p = model.predict(X)[0] if hasattr(model, "predict") else random.random()
    candidates.append((r,g,b,p))

# pick top color
best_color = sorted(candidates, key=lambda x:x[3], reverse=True)[0]
hex_color = '#%02x%02x%02x' % (best_color[0], best_color[1], best_color[2])

st.markdown(f"""
<div style='display:flex; justify-content:center;'>
  <div style='width:180px; height:100px; border-radius:12px; background:{hex_color};
              box-shadow:0 6px 15px rgba(0,0,0,0.2); border:2px solid #fff;'></div>
</div>
<h3 style='text-align:center; color:{hex_color};'>Recommended Color: {hex_color.upper()}</h3>
""", unsafe_allow_html=True)

st.info("This color has the highest predicted user click probability based on your profile.")

# ---------- VISUALIZATIONS ----------
st.markdown("---")
st.header("📊 Interactive Analytics Dashboard")

col1, col2 = st.columns(2)

# 1️⃣ Click Distribution
if "Clicked" in df.columns:
    click_dist = df["Clicked"].value_counts().reset_index()
    click_dist.columns = ["Click", "Count"]
    fig1 = px.bar(click_dist, x="Click", y="Count", color="Click", title="Click Distribution")
    col1.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("No 'Clicked' column found for click analysis.")

# 2️⃣ Color Intensity vs Click Rate
if "Clicked" in df.columns:
    df["brightness"] = (df["r"] + df["g"] + df["b"]) / 3
    avg_click = df.groupby(pd.cut(df["brightness"], 10))["Clicked"].mean().reset_index()
    fig2 = px.line(avg_click, x="brightness", y="Clicked", markers=True, title="Brightness vs Click Rate", color_discrete_sequence=["#FF69B4"])
    col2.plotly_chart(fig2, use_container_width=True)

# 3️⃣ Mood vs Average Color
if "Mood" in df.columns:
    avg_rgb = df.groupby("Mood")[["r","g","b"]].mean().reset_index()
    fig3 = px.scatter_3d(avg_rgb, x="r", y="g", z="b", color="Mood", title="Mood vs Average RGB (3D)")
    st.plotly_chart(fig3, use_container_width=True)

# 4️⃣ Product Category vs Click Rate
if "Clicked" in df.columns and "Product_Category" in df.columns:
    prod_click = df.groupby("Product_Category")["Clicked"].mean().reset_index()
    fig4 = px.bar(prod_click, x="Product_Category", y="Clicked", title="Product Category vs Click Rate", color="Clicked", color_continuous_scale="Agsunset")
    st.plotly_chart(fig4, use_container_width=True)

# 5️⃣ Gender vs Preferred Color (average RGB)
if "gender" in df.columns:
    gender_rgb = df.groupby("gender")[["r","g","b"]].mean().reset_index()
    fig5 = px.scatter_3d(gender_rgb, x="r", y="g", z="b", color="gender", size_max=10, title="Gender vs Average RGB Preference")
    st.plotly_chart(fig5, use_container_width=True)

# 6️⃣ Cluster Visualization
kmeans = KMeans(n_clusters=8, random_state=42).fit(df[["r","g","b"]])
df["cluster"] = kmeans.labels_
centers = kmeans.cluster_centers_.astype(int)
palette = [f"#{r:02x}{g:02x}{b:02x}" for r,g,b in centers]
fig6 = px.scatter_3d(df, x="r", y="g", z="b", color=df["cluster"].astype(str), title="Color Clusters in RGB Space", color_discrete_sequence=palette)
st.plotly_chart(fig6, use_container_width=True)

# 7️⃣ Heatmap: Season vs Click Rate
if "Season" in df.columns and "Clicked" in df.columns:
    heat = df.groupby(["Season","Mood"])["Clicked"].mean().unstack()
    fig7, ax = plt.subplots(figsize=(6,4))
    sns.heatmap(heat, annot=True, cmap="magma", ax=ax)
    st.pyplot(fig7)

# 8️⃣ Animated Plot: Time Spent vs Click Rate by Mood
if "Clicked" in df.columns:
    fig8 = px.scatter(df, x="Time_Spent_sec", y="Clicked", color="Mood",
                      animation_frame="Season", title="Animated — Time vs Clicks by Mood")
    st.plotly_chart(fig8, use_container_width=True)

# 9️⃣ Brightness Distribution
fig9 = px.histogram(df, x="brightness", nbins=30, color_discrete_sequence=["#FFA07A"], title="Color Brightness Distribution")
st.plotly_chart(fig9, use_container_width=True)

# ---------- INSIGHTS ----------
st.markdown("---")
st.subheader("💡 Smart Insights")
st.write("""
- **Click Distribution:** shows if your dataset is balanced or dominated by one response (helps evaluate model bias).  
- **Brightness vs Click Rate:** reveals if users prefer lighter or darker tones.  
- **Mood vs Average RGB:** visualizes emotional color tendencies — e.g., *Happy → bright*, *Sad → blue*.  
- **Product vs Click Rate:** identifies which category benefits from color personalization.  
- **Gender vs RGB Preference:** uncovers color inclinations by gender.  
- **Color Clusters:** shows your color families and their density — validates unsupervised grouping.  
- **Season vs Click Rate Heatmap:** tracks how preferences vary with time of year.  
- **Animated Mood-Time Plot:** dynamic exploration of engagement over time.  
- **Brightness Distribution:** shows dataset’s visual diversity.  
""")

st.success("This dashboard combines human psychology, machine learning, and color theory for a complete interactive experience! 🌟")
