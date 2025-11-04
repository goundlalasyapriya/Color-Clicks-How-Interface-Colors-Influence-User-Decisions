import pandas as pd
s=pd.Series([8,24,56,43],index=['a','b','c','d'])
print(s)
print("value at index 'b': ",s['b'])

print("mean: ",s.mean())
import pandas as pd
df=pd.DataFrame({
'name':['lasyaa','priya'],
    'age':[8,16]
})
print(df)
df.head()

print("Head:\n", df.head())
print("Data types:\n", df.dtypes)
print("Shape:", df.shape)
print("info:",df.info())
print("index:",df.index)

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', None],
    'Age': [25, None, 30, 22, 28],
    'Salary': [50000, 60000, None, 52000, 58000],
    'Department': ['HR', 'IT', 'IT', None, 'Finance']
}
df=pd.DataFrame(data)
