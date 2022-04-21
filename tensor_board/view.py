import pandas as pd
import numpy as np


print("------ df_train ------")
df_train = pd.read_csv('./train.csv', nrows=1000, skiprows=1)
df_train.rename(columns = {'Unnamed: 0':'datetime', 'Unnamed: 1':'instrument'}, inplace = True)
df_train = df_train.iloc[1:]
print(df_train.shape)
print("CLOSE0: ", df_train.iloc[1]["CLOSE0"])
print(df_train)

print("------ df_valid ------")
df_valid = pd.read_csv('./valid.csv', nrows=1000, skiprows=1)
df_valid.rename(columns = {'Unnamed: 0':'datetime', 'Unnamed: 1':'instrument'}, inplace = True)
df_valid = df_train.iloc[1:]
print(df_valid.shape)
print(df_valid)


df_w = pd.DataFrame(np.ones_like(df_valid.values), index=df_valid.index)
print(df_w.shape)
print(df_w)