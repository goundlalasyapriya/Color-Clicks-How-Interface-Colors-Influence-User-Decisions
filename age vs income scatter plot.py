import matplotlib.pyplot as plt

import pandas as pd
data={
    'age':[25,32,47,51,62,23,45,36],
    'income':[50000,60000,80000,90000,75000,48000,67000,62000]

}
df=pd.DataFrame(data)
plt.scatter(data=df,x='age',y='income')
plt.title('age vs income')
plt.show()