import pandas as pd

# If you want to reduce the dataset
df = pd.read_parquet('logos.snappy.parquet')

# Change from 100 to how many entries you want to keep
df.head(100).to_parquet('red.snappy.parquet')
