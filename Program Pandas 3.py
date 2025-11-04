import pandas as pd
data = {
    'outlook': ['sunny', 'sunny', 'overcast', 'rainy', 'rainy', 'rainy',
                'overcast', 'sunny', 'sunny', 'rainy', 'sunny', 'overcast', 'overcast', 'rainy'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool',
                    'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal',
                 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'windy': [False, True, False, False, False, True,
              True, False, False, False, True, True, False, True],
    'play': ['no', 'no', 'yes', 'yes', 'yes', 'no',
             'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
#a)Load the dataset into a DataFrame
df=pd.DataFrame(data)
#b) Display the first and last few rows
print("\n=== Head(First 5 rows) ===")
print(df.head())
print("\n=== Tail(Last 5 rows) ===")
print(df.tail())
#c)Check the dimensions of the dataset
print("\n=== Dimensions ===")
print(f"Rows: {df.shape[0]},Columns : {df.shape[1]}")
#View column names
print("\n===  column names  ===")
print(df.columns.tolist())
#d)Get information about the dataset
print("\n===  Data info  ===")
print(df.info())
#Check for missing values

print("\n===  missing values  ===")
print(df.isnull().sum())
#g) Basic statistical information
print("\n === statistical Summary === ")
print(df.describe(include='all')) #all emoothe col statistical info is collected