import pandas as pd

# 🌟 Create a Series with custom index
s = pd.Series([18, 16, 24], index=['a', 'b', 'c'])
print(s)
print("index at a:", s['a'])

# 📋 Create first DataFrame with basic info
data = {
    'name': ['lasya', 'priya'],
    'age': [45, 645],
    'gender': ['f', 'f']
}
df1 = pd.DataFrame(data)
print(df1)
print("shape:", df1.shape)
print("datatypes:", df1.dtypes)

# 🎓 Create second DataFrame with student scores
df2 = pd.DataFrame({
    'class': ['a', 'a', 'b', 'b', 'c', 'c'],
    'Student': ['John', 'Jane', 'Dave', 'Dana', 'Mike', 'Mona'],
    'Math': [88, 92, 79, 85, 94, 90],
    'English': [75, 85, 80, 88, 90, 92]
})

# 🔍 Access column and rows before setting index
print(df2['Math'])         # Access Math column
print(df2.iloc[2])         # Access 3rd row by position
print(df2.loc[2, 'Student'])  # Access by label (before setting index)

# 📌 Set 'Student' as index
df2 = df2.set_index('Student')

# 🎯 Access row for student 'Dave'
print(df2.loc['Dave'])     # Access full row for 'Dave'

# 📚 Access English scores and class (Student is now index)
print(df2.loc[:, ['English', 'class']])

# 📊 Sort by Math scores (descending)
df2_sorted = df2.sort_values('Math', ascending=False)

# 🔁 Reset index to bring 'Student' back as a column
df2_sorted = df2_sorted.reset_index()

# ✅ Display Student and Math columns from sorted DataFrame
print(df2_sorted[['Student', 'Math']])
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]
y = [10, 20, 25, 30]

plt.plot(x, y, color='purple', marker='o')
plt.title("Simple Line Plot")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

import numpy as np

# Creating an array
data = np.array([10, 20, 30, 40])

# Performing operations
print("Mean:", np.mean(data))
print("Standard Deviation:", np.std(data))
print("Array + 5:", data + 5)

