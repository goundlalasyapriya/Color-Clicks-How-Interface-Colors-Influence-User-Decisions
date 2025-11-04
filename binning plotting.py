import pandas as pd
import matplotlib.pyplot as plt

scores = [55, 67, 45, 90, 88, 76, 59, 62, 79, 85, 70, 85, 70, 66, 58, 80, 92]
df = pd.DataFrame({'Scores': scores})

bins = [40, 60, 80, 100]
labels = ['40-59', '60-79', '80-99']
df['Binned'] = pd.cut(df['Scores'], bins=bins, labels=labels, right=False)

df['Binned'].value_counts().sort_index().plot(kind='bar')
plt.xlabel('Score Ranges')
plt.ylabel('Count')  # Optional: adds clarity
plt.title('Score Distribution')  # Optional: adds clarity
plt.show()  # <-- This line is essential to display the plot