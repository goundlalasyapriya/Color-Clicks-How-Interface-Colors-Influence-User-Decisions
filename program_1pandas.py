import pandas as pd

# Fix file path
df = pd.read_csv(r"C:\DAV PROGRAM\employee_salary_dataset.csv")

# Show last and first rows
print(df.tail())
print(df.head())

# Salary column type
print(df['salary'].dtype)

# Gender column
print(df['gender'])

# Gender and Salary columns
print(df[['gender','salary']])

# Group by rank (mean of all numeric columns)
df_rank = df.groupby('rank').mean(numeric_only=True)
print(df_rank)

# Group by rank (mean salary only)
print(df.groupby('rank')[['salary']].mean())

# Subset: employees with salary > 120000
df_sub = df[df['salary'] > 120000]
print(df_sub)

# Subset: only females
df_f = df[df['gender'] == 'Female']
print(df_f)

# Print salary column
print(df['salary'])

# Rank + Salary
print(df[['rank','salary']])

# Slice rows 4–8
print(df[4:9])

# Rows 1–10, only rank, gender, salary
print(df.loc[1:10, ['rank','gender','salary']])

# Row by index (example: first row)
print(df.iloc[0])

# Example with i
i = 2
print(df.iloc[i])