import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve", None],
    "Age": [25, 30, None, 40, 22, 28],
    "Salary": [40000, 42000, 43000, 1000000, 45000, 47000],
    "Department": ["HR", "IT", "Finance", "IT", None, "HR"]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

print("\n(a) Dataset Exploration:\n")
print(df.head(), "\n")
df.info()
print("\nSummary statistics:\n", df.describe(include='all'))

df['Name'] = df['Name'].fillna(df['Name'].mode()[0])
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])
print("\n(b) After handling missing values:\n", df)

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df['Salary'] = np.where(df['Salary'] > upper_bound, upper_bound,
                        np.where(df['Salary'] < lower_bound, lower_bound, df['Salary']))
print("\n(c) After treating outliers:\n", df)

scaler = MinMaxScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])
print("\n(d) After Min-Max Normalization:\n", df)

encoder = LabelEncoder()
df['Department_encode'] = encoder.fit_transform(df['Department'])
print("\n(e) After Encoding Department:\n", df)
