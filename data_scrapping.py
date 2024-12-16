import pandas as pd 

df = pd.read_csv("training_dataset.csv")

for data in df.columns:
    print(f"{data}-> datatype: {df[data].dtypes}")