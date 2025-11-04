import pandas as pd
import numpy as np

# =========================
# Step 1: Load datasets
# =========================
ad_clicks_path = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\ad_click_dataset.csv"
color_path = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\color_names.csv"

ad_df = pd.read_csv(ad_clicks_path)
color_df = pd.read_csv(color_path)

print("✅ Ad Clicks Columns:", list(ad_df.columns))
print("✅ Color Columns:", list(color_df.columns))
print()

# =========================
# Step 2: Rename color dataset columns properly
# =========================
color_df = color_df.rename(columns={
    'Name': 'color_name',
    'Hex (24 bit)': 'hex_code',
    'Red (8 bit)': 'r',
    'Green (8 bit)': 'g',
    'Blue (8 bit)': 'b'
})

# Keep only what we need
color_df = color_df[['color_name', 'hex_code', 'r', 'g', 'b']].dropna().reset_index(drop=True)

print("🎨 Cleaned color dataset columns:", list(color_df.columns))
print(color_df.head(), "\n")

# =========================
# Step 3: Randomly assign colors to ad click entries
# =========================
np.random.seed(42)
random_colors = color_df.sample(n=len(ad_df), replace=True).reset_index(drop=True)

# Combine datasets
combined = pd.concat([ad_df.reset_index(drop=True), random_colors], axis=1)

# =========================
# Step 4: Add synthetic behavioral columns
# =========================
moods = ['Happy', 'Sad', 'Calm', 'Angry', 'Neutral']
seasons = ['Summer', 'Winter', 'Monsoon', 'Spring', 'Autumn']
products = ['Tech', 'Fashion', 'Food', 'Home_Decor', 'Sports']

combined['Mood'] = np.random.choice(moods, size=len(combined))
combined['Season'] = np.random.choice(seasons, size=len(combined))
combined['Product_Category'] = np.random.choice(products, size=len(combined))
combined['Time_Spent_sec'] = np.random.randint(5, 300, size=len(combined))

# =========================
# Step 5: Reorder & Save
# =========================
cols_base = [c for c in ['id', 'full_name', 'age', 'gender', 'device_type', 'ad_position'] if c in combined.columns]

final_cols = cols_base + [
    'Product_Category', 'color_name', 'hex_code', 'r', 'g', 'b',
    'Mood', 'Season', 'Time_Spent_sec'
]

if 'click' in combined.columns:
    final_cols.append('click')

combined = combined[final_cols]

# =========================
# Step 6: Save augmented dataset
# =========================
output_path = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\augmented_color_click_dataset.csv"
combined.to_csv(output_path, index=False)

print("✅ Augmented dataset created successfully!")
print(f"📁 Saved at: {output_path}")
print("📊 Rows:", combined.shape[0], " | Columns:", combined.shape[1])
print("\n✨ Sample of Final Dataset:")
print(combined.head(10))
