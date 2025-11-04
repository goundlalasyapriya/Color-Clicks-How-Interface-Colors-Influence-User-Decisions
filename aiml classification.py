# ---------------------------------------------
# Netflix Dataset Classification Pipeline
# ---------------------------------------------

# 1️⃣ Import all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, classification_report

# ---------------------------------------------
# 2️⃣ Load the Dataset
# ---------------------------------------------
path = r"C:\Users\LASYA PRIYA\PycharmProjects\ProgramPandas\netflix_titles.csv"
df = pd.read_csv(path)
print("Dataset loaded successfully ✅")
print(df.head())
print("\nShape:", df.shape)
print("\nColumns:", df.columns.tolist())

# ---------------------------------------------
# 3️⃣ Check type of dataset
# ---------------------------------------------
print("\nType of Data: CSV file")
print("Labelled Dataset ✅ (Target column = 'type' -> Movie or TV Show)")
print("We will check balance of target class:")

print(df['type'].value_counts())
sns.countplot(x='type', data=df)
plt.title("Distribution of Target (Movies vs TV Shows)")
plt.show()

# ---------------------------------------------
# 4️⃣ Preprocessing
# ---------------------------------------------

# Handle missing values
df['rating'].fillna("Not Rated", inplace=True)
df['country'].fillna("Unknown", inplace=True)
df['duration'].fillna("0", inplace=True)
df['description'].fillna("No Description", inplace=True)
print(df)
# Feature engineering
def extract_duration(x):
    import re
    m = re.search(r'(\d+)', str(x))
    return float(m.group(1)) if m else 0

df['duration_num'] = df['duration'].apply(extract_duration)
df['desc_len'] = df['description'].apply(len)
df['num_genres'] = df['listed_in'].fillna('').apply(lambda x: len([g for g in x.split(',') if g.strip()]))

# ---------------------------------------------
# 5️⃣ Feature Selection / Extraction
# ---------------------------------------------
top_countries = df['country'].value_counts().head(10).index
df['country_top'] = df['country'].apply(lambda x: x if x in top_countries else 'Other')

genre_series = df['listed_in'].dropna().str.get_dummies(sep=',')
top_genres = genre_series.sum().sort_values(ascending=False).head(8).index.tolist()
for g in top_genres:
    df['genre_' + g.strip().replace(' ', '_')] = df['listed_in'].fillna('').apply(lambda x: 1 if g in x else 0)

features = ['release_year','duration_num','desc_len','num_genres','country_top','rating'] + [f'genre_{g.strip().replace(" ","_")}' for g in top_genres]

df_model = df[features + ['type']].dropna()
print("\nFeature Columns Used:", features)

# Encode categorical columns
df_model = pd.get_dummies(df_model, columns=['country_top','rating'], drop_first=True)
le = LabelEncoder()
df_model['type_label'] = le.fit_transform(df_model['type'])

X = df_model.drop(['type','type_label'], axis=1)
y = df_model['type_label']

# Scale numeric features
scaler = MinMaxScaler()
X[X.columns] = scaler.fit_transform(X[X.columns])

# ---------------------------------------------
# 6️⃣ Check Dataset Balance
# ---------------------------------------------
print("\nTarget class distribution:")
print(y.value_counts())

sns.countplot(x=y)
plt.title("Class Distribution After Preprocessing")
plt.show()

# ---------------------------------------------
# 7️⃣ Train-Test Split
# ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ---------------------------------------------
# 8️⃣ Cross Validation & Algorithm Selection
# ---------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
log_model = LogisticRegression(max_iter=500)
rf_model = RandomForestClassifier(random_state=42, n_estimators=100)

print("\nCross-validation F1 scores:")
print("Logistic Regression:", cross_val_score(log_model, X_train, y_train, cv=cv, scoring='f1').mean())
print("Random Forest:", cross_val_score(rf_model, X_train, y_train, cv=cv, scoring='f1').mean())

# ---------------------------------------------
# 9️⃣ Train & Evaluate Model
# ---------------------------------------------
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:,1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------------------------------------------
# 🔟 ROC Curve
# ---------------------------------------------
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1],'--',color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Random Forest")
plt.legend()
plt.show()

# ---------------------------------------------
# 1️⃣1️⃣ Optimization (Hyperparameter Tuning)
# ---------------------------------------------
param_grid = {'n_estimators':[100,200], 'max_depth':[None,10,20]}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
best_model = grid.best_estimator_

# Evaluate optimized model
y_pred_best = best_model.predict(X_test)
print("\nAccuracy After Optimization:", accuracy_score(y_test, y_pred_best))

# Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
importances.plot.barh()
plt.title("Top 15 Important Features")
plt.show()
