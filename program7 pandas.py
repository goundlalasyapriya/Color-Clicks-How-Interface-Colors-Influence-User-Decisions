import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve", None],
    "Age": [25, 30, None, 40, 22, 28],
    "Salary": [40000, 42000, 43000, 1000000, 45000, 47000],
    "Department": ["HR", "IT", "Finance", "IT", None, "HR"]
}

df = pd.DataFrame(data)

print(df.head())
print(df.info())
print(df.describe())

df_cleaned = df.dropna()

df['Age'].fillna(df['Age'].mean())
df['Department'].fillna(df['Department'].mode()[0])
df['Name'].fillna("Unknown")

Q1=df['Salary'].quantile(0.25)
Q3=df['Salary'].quantile(0.75)
IQR=Q3-Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Lower bound: {lower_bound}, Upper bound: {upper_bound}")

outliers=df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print("\nOutliers detected:")
print(outliers)

def cap_value(x):
    if x > upper_bound:
        return upper_bound
    elif x < lower_bound:
        return lower_bound
    else:
        return x

df['Salary']=df['Salary'].apply(cap_value)

print("\nData after capping outliers:")
print(df)

scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

le = LabelEncoder()
df['Department_encoded'] = le.fit_transform(df['Department'])

print("\nAfter encoding Department:\n", df[['Department','Department_encoded']])

print("\nFinal DataFrame:\n",df)