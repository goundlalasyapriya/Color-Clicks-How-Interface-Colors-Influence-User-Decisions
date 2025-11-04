import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample Data
np.random.seed(42)
data = pd.DataFrame({
    'Category': np.random.choice(['A', 'B', 'C', 'D'], 30),
    'Values': np.random.randint(10, 100, 30),
    'Values2': np.random.randint(20, 120, 30),
    'Scores': np.random.normal(70, 10, 30)
})

# b) Line Chart - Mean of 'Values' by Category
data.groupby('Category')['Values'].mean().plot(marker='o', color='royalblue', title="Mean of 'Values' by Category", grid=True)
plt.xlabel("Category"); plt.ylabel("Mean Values"); plt.show()

# c) Bar Plot - Average 'Values2' by Category
data.groupby('Category')['Values2'].mean().plot(kind='bar', color='seagreen', title="Average 'Values2' by Category", grid=True)
plt.xlabel("Category"); plt.ylabel("Average Values2"); plt.show()

# d) Scatter Plot - 'Values' vs 'Values2'
plt.scatter(data['Values'], data['Values2'], color='darkviolet', edgecolors='black')
plt.title("Scatter Plot: Values vs Values2"); plt.xlabel("Values"); plt.ylabel("Values2"); plt.grid(True); plt.show()

# e) Histogram - Distribution of 'Scores'
plt.hist(data['Scores'], bins=8, color='orange', edgecolor='black')
plt.title("Histogram of 'Scores'"); plt.xlabel("Scores"); plt.ylabel("Frequency"); plt.grid(axis='y'); plt.show()

# f) Box Plot - 'Scores'
sns.boxplot(y=data['Scores'], color='lightgreen')
plt.title("Box Plot of 'Scores'"); plt.ylabel("Scores"); plt.grid(True, axis='y'); plt.show()

# g) Heatmap - Correlation Matrix
sns.heatmap(data[['Values', 'Values2', 'Scores']].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap"); plt.show()

# h) Pie Chart - Category Distribution
plt.pie(data['Category'].value_counts(), labels=data['Category'].value_counts().index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title("Category Percentage Distribution"); plt.show()
