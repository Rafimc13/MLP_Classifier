import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import tarfile
from sklearn.datasets import load_files



movies_train = load_files(container_path="aclImdb/train", encoding="utf-8")
movies_test = load_files(container_path="aclImdb/test", encoding="utf-8")

# Transform train dataset into a dataframe
data_train = {'reviews': movies_train.data, 'sentiment': movies_train.target}
df_train = pd.DataFrame(data_train)

# Transform test dataset into a dataframe
data_test = {'reviews': movies_test.data, 'sentiment': movies_test.target}
df_test = pd.DataFrame(data_test)

print(df_train.iloc[:10])
print('----------------------------------------------------------------------------------------')
print(df_test.iloc[:10])

# Drop rows where 'sentiment' is 2
df_train = df_train[df_train['sentiment'] != 2]
df_test = df_test[df_test['sentiment'] != 2]

print(len(df_train))
print(len(df_test))

df_test = df_test.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)

df_train = pd.concat([df_train, df_test[:5000]])
