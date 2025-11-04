import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

# ----------------- Step 1: Load datasets -----------------
df_shopping_behavior = pd.read_csv(r"C:\DAV PROGRAM\shopping_behavior_updated.csv")
df_shopping_trends = pd.read_csv(r"C:\DAV PROGRAM\shopping_trends.csv")
df_ab_data = pd.read_csv(r"C:\DAV PROGRAM\ab_data.csv")

print("\nStep 1: Loaded DataFrames")
print("Shopping Behavior Data:")
print(df_shopping_behavior.head())
print("Shopping Trends Data:")
print(df_shopping_trends.head())
print("A/B Test Data:")
print(df_ab_data.head())

# Strip spaces in column names
df_shopping_behavior.columns = df_shopping_behavior.columns.str.strip()
df_shopping_trends.columns = df_shopping_trends.columns.str.strip()
df_ab_data.columns = df_ab_data.columns.str.strip()

print("\nStep 1b: After Stripping Column Names")
print("Shopping Behavior Columns:", df_shopping_behavior.columns.tolist())
print("Shopping Trends Columns:", df_shopping_trends.columns.tolist())
print("A/B Columns:", df_ab_data.columns.tolist())

# Rename Payment Method if exists
if 'Preferred Payment Method' in df_shopping_trends.columns:
    df_shopping_trends.rename(columns={'Preferred Payment Method': 'Payment Method'}, inplace=True)

print("\nStep 1c: After Renaming Columns in Shopping Trends")
print(df_shopping_trends.head())

# ----------------- Step 2: Merge shopping datasets -----------------
merged_shopping_df = pd.merge(
    df_shopping_behavior,
    df_shopping_trends,
    on='Customer ID',
    how='outer',
    suffixes=('_behavior', '_trends')
)

print("\nStep 2: After Merging Shopping Behavior & Trends")
print(merged_shopping_df.head())

# ----------------- Step 3: Merge duplicate columns safely -----------------
for col in df_shopping_behavior.columns:
    col_behavior = f'{col}_behavior'
    col_trends = f'{col}_trends'

    if col_behavior in merged_shopping_df.columns and col_trends in merged_shopping_df.columns:
        merged_shopping_df[col] = merged_shopping_df.apply(
            lambda row: row[col_behavior] if pd.notna(row[col_behavior]) else row[col_trends],
            axis=1
        )
        merged_shopping_df.drop(columns=[col_behavior, col_trends], inplace=True)

print("\nStep 3: After Handling Duplicate Columns")
print(merged_shopping_df.head())

# ----------------- Step 4: Handle missing Payment Method -----------------
if 'Payment Method' in merged_shopping_df.columns:
    merged_shopping_df.loc[merged_shopping_df['Customer ID'] == 52, 'Payment Method'] = 'Credit Card'

print("\nStep 4: After Handling Missing Payment Method")
print(merged_shopping_df.head())

# ----------------- Step 5: Fill missing values -----------------
num_cols = merged_shopping_df.select_dtypes(include=[np.number]).columns
merged_shopping_df[num_cols] = merged_shopping_df[num_cols].fillna(merged_shopping_df[num_cols].median())

cat_cols = merged_shopping_df.select_dtypes(include=['object']).columns
merged_shopping_df[cat_cols] = merged_shopping_df[cat_cols].fillna('Unknown')

print("\nStep 5: After Filling Missing Values")
print(merged_shopping_df.head())

# ----------------- Step 6: One-hot encode categorical columns -----------------
categorical_columns = merged_shopping_df.select_dtypes(include=['object']).columns.tolist()
if 'Customer ID' in categorical_columns:
    categorical_columns.remove('Customer ID')

df_shopping_preprocessed = pd.get_dummies(
    merged_shopping_df,
    columns=categorical_columns,
    drop_first=True
)

print("\nStep 6: After One-Hot Encoding")
print(df_shopping_preprocessed.head())

# ----------------- Step 7: Scale numeric columns -----------------
scaler = StandardScaler()
df_shopping_preprocessed[num_cols] = scaler.fit_transform(df_shopping_preprocessed[num_cols])

print("\nStep 7: After Scaling Numeric Columns")
print(df_shopping_preprocessed.head())

# ----------------- Step 8: Preprocess A/B dataset -----------------
if 'timestamp' in df_ab_data.columns:
    df_ab_data['timestamp'] = pd.to_datetime(df_ab_data['timestamp'], errors='coerce')

print("\nStep 8a: After Converting Timestamp")
print(df_ab_data.head())

if 'group' in df_ab_data.columns and 'landing_page' in df_ab_data.columns:
    df_ab_data_cleaned = df_ab_data[
        ((df_ab_data['group'] == 'control') & (df_ab_data['landing_page'] == 'old_page')) |
        ((df_ab_data['group'] == 'treatment') & (df_ab_data['landing_page'] == 'new_page'))
    ]
else:
    df_ab_data_cleaned = df_ab_data.copy()
    print("Warning: 'group' or 'landing_page' missing in A/B dataset")

print("\nStep 8a: After Filtering A/B Dataset")
print(df_ab_data_cleaned.head())

if 'timestamp' in df_ab_data_cleaned.columns:
    df_ab_data_cleaned['year'] = df_ab_data_cleaned['timestamp'].dt.year
    df_ab_data_cleaned['month'] = df_ab_data_cleaned['timestamp'].dt.month
    df_ab_data_cleaned['day'] = df_ab_data_cleaned['timestamp'].dt.day
    df_ab_data_cleaned['weekday'] = df_ab_data_cleaned['timestamp'].dt.weekday

print("\nStep 8b: After Feature Engineering")
print(df_ab_data_cleaned.head())

num_cols_ab = df_ab_data_cleaned.select_dtypes(include=[np.number]).columns
df_ab_data_cleaned[num_cols_ab] = df_ab_data_cleaned[num_cols_ab].fillna(df_ab_data_cleaned[num_cols_ab].median())

cat_cols_ab = df_ab_data_cleaned.select_dtypes(include=['object']).columns
df_ab_data_cleaned[cat_cols_ab] = df_ab_data_cleaned[cat_cols_ab].fillna('Unknown')

print("\nStep 8c: After Filling Missing Values in A/B Data")
print(df_ab_data_cleaned.head())

categorical_cols_ab = ['group', 'landing_page']
categorical_cols_ab = [col for col in categorical_cols_ab if col in df_ab_data_cleaned.columns]
df_ab_data_cleaned = pd.get_dummies(df_ab_data_cleaned, columns=categorical_cols_ab, drop_first=True)

print("\nStep 8d: After Encoding A/B Categorical Columns")
print(df_ab_data_cleaned.head())

scaler_ab = StandardScaler()
df_ab_data_cleaned[num_cols_ab] = scaler_ab.fit_transform(df_ab_data_cleaned[num_cols_ab])

print("\nStep 8e: After Scaling A/B Numeric Columns")
print(df_ab_data_cleaned.head())

print("\nFinal A/B Data Info:")
print(df_ab_data_cleaned.info())

# ----------------- Step 9: Save preprocessed data to CSV -----------------
os.makedirs(r"C:\DAV PROGRAM", exist_ok=True)

df_shopping_preprocessed.to_csv(r"C:\DAV PROGRAM\shopping_transformed.csv", index=False)

df_ab_data_cleaned.to_csv(r"C:\DAV PROGRAM\ab_transformed.csv", index=False)

print("\nStep 9: Transformed datasets saved successfully as:")
print("Shopping Data: C:\\DAV PROGRAM\\shopping_transformed.csv")
print("A/B Data: C:\\DAV PROGRAM\\ab_transformed.csv")
