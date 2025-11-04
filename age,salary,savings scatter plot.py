import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data={
    'age':[25,32,47,51,62,23,45,36],
    'income':[50000,60000,80000,90000,75000,48000,67000,62000],
    'savings':[20000,15000,40000,30000,35000,10000,25000,22000]
}
df=pd.DataFrame(data)

sns.scatterplot(
    x='age',
    y='income',
    size='savings',
    data=df,
    hue='savings',
    palette='coolwarm',
    sizes=(20, 600)
)

plt.title('Age vs Income with Savings as size and color')
plt.xlabel('Age')
plt.ylabel('Income($)')
plt.legend(title='Savings')

plt.show()