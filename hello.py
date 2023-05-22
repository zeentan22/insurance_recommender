# Import pandas library
import pandas as pd

# initialize list elements
data = [10,20,30,40,50,60]

# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['Numbers'])


def condition(row):
    if row.Numbers > 30:
        return True
    else:
        return False


def filter(condition, df):
    if len(df.index) > 0:
        df = df[df.apply(condition, axis=1)]
    return df
    

# print dataframe.
print(df)
df = filter(condition,df)
print(type(df))

