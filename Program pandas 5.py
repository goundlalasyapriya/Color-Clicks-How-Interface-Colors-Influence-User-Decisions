import pandas as pd
data = {'Salary': [40000, 42000, 43000, 44000, 45000, 1000000]}
df = pd.DataFrame(data)
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3-Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
print(f"Lower bound : {lower_bound}, Upper bound : {upper_bound}")
# step 2
outliers = df[(df['Salary'] < lower_bound)  |  (df['Salary'] > upper_bound)]
print("\nOutliers detected: ")
print(outliers)
# step 3


def cap_value(x):
    if x > upper_bound:
        return upper_bound
    elif x < lower_bound:
        return lower_bound
    else:
        return x
df['Salary'] = df['Salary'].apply(cap_value)
print("\nData after capping outliers:")
print(df)
