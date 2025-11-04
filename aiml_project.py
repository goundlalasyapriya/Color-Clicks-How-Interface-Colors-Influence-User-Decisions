import pandas as pd

# Load datasets
df_medai = pd.read_csv(r"C:\Users\LASYA PRIYA\Downloads\aiml_project\MedAI.csv")
df_symptoms = pd.read_csv(r"C:\Users\LASYA PRIYA\Downloads\aiml_project\DiseaseAndSymptoms.csv")

# ==========================
# 🧹 1. Clean MedAI.csv
# ==========================
df_medai.columns = df_medai.columns.str.strip()
df_medai['Disease'] = df_medai['Disease'].str.strip().str.lower()
df_medai = df_medai.drop_duplicates()

# ==========================
# 🧼 2. Clean DiseaseAndSymptoms.csv
# ==========================
df_symptoms.columns = df_symptoms.columns.str.strip()

for col in df_symptoms.columns:
    df_symptoms[col] = df_symptoms[col].astype(str).str.strip().str.lower()

df_symptoms.replace(['nan', 'none', ''], 'none', inplace=True)
df_symptoms = df_symptoms.drop_duplicates()

# ==========================
# ✅ Save transformed datasets
# ==========================
df_medai.to_csv(r"C:\Users\LASYA PRIYA\Downloads\aiml_project\MedAI_transformed.csv", index=False)
df_symptoms.to_csv(r"C:\Users\LASYA PRIYA\Downloads\aiml_project\DiseaseAndSymptoms_transformed.csv", index=False)


print("✅ Preprocessing complete!")
print("\n--- Cleaned MedAI.csv Preview ---")
print(df_medai.head())

print("\n--- Cleaned DiseaseAndSymptoms.csv Preview ---")
print(df_symptoms.head())
