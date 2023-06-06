import pandas as pd

df = pd.read_csv('train.csv')
subset_df = df.sample(n=int(len(df) * 0.1))
subset_df.to_csv('train_subset.csv', index=False)
