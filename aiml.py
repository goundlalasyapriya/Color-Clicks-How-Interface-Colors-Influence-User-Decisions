import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)
from imblearn.over_sampling import RandomOverSampler

# ---------------------------------------------
# 1️⃣ TYPE OF DATASET
# ---------------------------------------------
print("\n--------------------------------------------")
print("1️⃣ TYPE OF DATASET")
print("--------------------------------------------")

df = pd.read_csv("netflix_titles.csv")
print("✅ Dataset loaded successfully!")
print("Shape of dataset:", df.shape)
print("Columns:", df.columns.tolist())

print("\n📘 Type of Data: CSV file")
print("📘 Type of Dataset: Labeled (has 'type' column as label)")
print("📘 Feature Types: Text (description), Categorical (country, rating), Numerical (release_year)")
print("\nDistribution of Target Labels:")
print(df["type"].value_counts())

sns.countplot(x="type", data=df)
plt.title("Before Balancing: Target Distribution")
plt.tight_layout()
plt.show()

print("\n📌 Step 1 Summary:")
print("✔️ Type of Data: CSV file")
print("✔️ Dataset is Labeled (target column: 'type')")
print("✔️ Initial Distribution: Imbalanced (more Movies than TV Shows)")
print("✔️ Feature Types: Text, Categorical, Numerical")

# ---------------------------------------------
# 2️⃣ PREPROCESSING TECHNIQUES
# ---------------------------------------------
df = df.fillna({
    "rating": "Unknown",
    "country": "Unknown",
    "listed_in": "Unknown",
    "description": "",
    "duration": "0 min"
})
print("✅ Missing values handled using fillna()")
print("Remaining missing values per column:\n", df.isnull().sum())

print("\n📌 Step 2 Summary:")
print("✔️ Text: Filled missing descriptions with empty string")
print("✔️ Categorical: Filled missing 'rating', 'country', 'listed_in' with 'Unknown'")
print("✔️ Numerical: 'duration' filled with '0 min' to support regression")

# ---------------------------------------------
# 3️⃣ BALANCING THE DATASET
# ---------------------------------------------
print("\n--------------------------------------------")
print("3️⃣ BALANCING THE DATASET")
print("--------------------------------------------")

X = df[["description", "rating", "country", "release_year"]]
y = df["type"]

ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, y)
print("✅ Dataset balanced using RandomOverSampler")
print("After balancing:\n", y_bal.value_counts())

sns.countplot(x=y_bal.values)
plt.title("After Balancing: Target Distribution")
plt.tight_layout()
plt.show()

print("\n📌 Step 3 Summary:")
print("✔️ Used RandomOverSampler to balance 'type' column")
print("✔️ Achieved equal representation of Movies and TV Shows")

# ---------------------------------------------
# 4️⃣ CROSS-VALIDATION & FEATURE EXTRACTION
# ---------------------------------------------
print("\n--------------------------------------------")
print("4️⃣ CROSS-VALIDATION TECHNIQUE USED")
print("--------------------------------------------")
print("✅ Using Stratified K-Fold Cross Validation (3 folds)")

print("\n4.1️⃣ FEATURE SELECTION / EXTRACTION")
print("✅ Using TF-IDF for text features, OneHotEncoder for categories, StandardScaler for numeric")

text_pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=300, stop_words="english"))])
cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
num_pipeline = Pipeline([("scaler", StandardScaler())])

preprocess = ColumnTransformer([
    ("text", text_pipeline, "description"),
    ("cat", cat_pipeline, ["rating", "country"]),
    ("num", num_pipeline, ["release_year"])
])

print("\n📌 Step 4 Summary:")
print("✔️ Cross-validation: Stratified K-Fold (3 folds)")
print("✔️ Feature Extraction: TF-IDF, OneHotEncoder, StandardScaler")

# ---------------------------------------------
# 5️⃣ TRAIN-TEST SPLIT
# ---------------------------------------------
print("\n--------------------------------------------")
print("5️⃣ TRAIN-TEST SPLIT")
print("--------------------------------------------")

X_train, X_test, y_train, y_test = train_test_split(
    X_bal, y_bal, test_size=0.2, stratify=y_bal, random_state=42
)
print(f"✅ Split Done: Train={X_train.shape}, Test={X_test.shape}")
print("Ratio used: 80% Train / 20% Test")

print("\n📌 Step 5 Summary:")
print("✔️ Train-Test Split: 80/20")
print("✔️ Stratified sampling preserves label distribution")

# ---------------------------------------------
# 6️⃣ ALGORITHM USED
# ---------------------------------------------
print("\n--------------------------------------------")
print("6️⃣ ALGORITHM USED")
print("--------------------------------------------")
print("✅ RandomForestClassifier (Ensemble Model) for Classification Task")

clf = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestClassifier(n_estimators=50, random_state=42))
])

print("\n📌 Step 6 Summary:")
print("✔️ Classification Algorithm: RandomForestClassifier")
print("✔️ Integrated with preprocessing pipeline")

# ---------------------------------------------
# 7️⃣ TRAIN AND TEST THE MODEL
# ---------------------------------------------
print("\n--------------------------------------------")
print("7️⃣ TRAIN AND TEST THE MODEL")
print("--------------------------------------------")

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
print("Cross-validation accuracy:", np.round(scores, 3))
print("Mean Accuracy:", np.round(scores.mean(), 3))

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\n✅ Model Trained Successfully!")

print("\n📌 Step 7 Summary:")
print("✔️ Model trained using pipeline")
print("✔️ Cross-validation accuracy printed")
print("✔️ Predictions generated on test set")

# ---------------------------------------------
# 8️⃣ PERFORMANCE METRICS - CLASSIFICATION
# ---------------------------------------------
print("\n--------------------------------------------")
print("8️⃣ PERFORMANCE METRICS - CLASSIFICATION")
print("--------------------------------------------")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap="Blues")
plt.title("Confusion Matrix (RandomForest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

print("\n📌 Step 8 Summary:")
print("✔️ Classification Report: Precision, Recall, F1-score")
print("✔️ Confusion Matrix visualizes prediction accuracy")
