import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Load Datasets safely
# -----------------------------
def load_csv(path):
    """Check if file exists before loading"""
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return None
    print(f"✅ Loaded: {path}")
    return pd.read_csv(path)

# Update these paths based on your actual file names
df_screen = load_csv(r"C:\DAV PROGRAM\impact_of_screen_time.csv")
df_teen = load_csv(r"C:\DAV PROGRAM\teen_smartphone_addiction.csv")
df_students = load_csv(r"C:\DAV PROGRAM\students_health_perf.csv")

# Stop if any dataset missing
if df_screen is None or df_teen is None or df_students is None:
    print("⚠️ One or more files are missing. Please check file names/locations.")
    exit()

# -----------------------------
# Step 2: Clean Column Names
# -----------------------------
def clean_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

df_screen = clean_columns(df_screen)
df_teen = clean_columns(df_teen)
df_students = clean_columns(df_students)

# -----------------------------
# Step 3: Handle Missing Values
# -----------------------------
def handle_missing(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    return df

df_screen = handle_missing(df_screen)
df_teen = handle_missing(df_teen)
df_students = handle_missing(df_students)

# -----------------------------
# Step 4: Encode Categorical Variables
# -----------------------------
def encode_categorical(df):
    cat_cols = df.select_dtypes(include=['object']).columns
    if 'id' in cat_cols:
        cat_cols = cat_cols.drop('id')  # don't encode ID column
    if len(cat_cols) > 0:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df

df_screen = encode_categorical(df_screen)
df_teen = encode_categorical(df_teen)
df_students = encode_categorical(df_students)

# -----------------------------
# Step 5: Normalize Numerical Variables
# -----------------------------
scaler = StandardScaler()

def scale_numeric(df):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df

df_screen = scale_numeric(df_screen)
df_teen = scale_numeric(df_teen)
df_students = scale_numeric(df_students)

# -----------------------------
# Step 6: Save Transformed Datasets
# -----------------------------
df_screen.to_csv("screen_time_transformed.csv", index=False)
df_teen.to_csv("teen_smartphone_transformed.csv", index=False)
df_students.to_csv("students_perf_transformed.csv", index=False)

print("\n✅ Transformed datasets saved successfully:")
print(" - screen_time_transformed.csv")
print(" - teen_smartphone_transformed.csv")
print(" - students_perf_transformed.csv")
