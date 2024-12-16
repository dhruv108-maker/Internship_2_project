import pandas as pd
from config import file_path

# Load the dataset
df = pd.read_excel(file_path)

# # print(df.columns)
# for c,t in zip(df.dtypes, df.columns):
#     print(f"{c}:{t}")
