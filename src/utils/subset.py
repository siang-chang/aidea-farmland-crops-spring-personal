# %%
import numpy as np
import pandas as pd

# %%
dataset = pd.read_csv('../data/label.csv')
dataset["kfold"] = np.nan
dataset["hafe"] = np.nan
dataset.groupby(['label', 'month']).apply(lambda x: dataset["kfold"].fillna(pd.Series(data=1, index=x.sample(frac=0.333333333333333, random_state=5397).index), inplace=True))
# dataset.groupby(['label', 'month']).apply(lambda x: dataset["hafe"].fillna(pd.Series(data=2, index=x[x["kfold"].notna()].sample(frac=0.5, random_state=5397).index), inplace=True))
dataset.groupby(['label', 'month']).apply(lambda x: dataset["kfold"].fillna(pd.Series(data=2, index=x[x["kfold"].isna()].sample(frac=0.5, random_state=5397).index), inplace=True))
dataset["kfold"].fillna(3, inplace=True)
# kfold = hafe or kfole
# dataset['kfold'] = dataset.apply(lambda x: x['kfold'] if np.isnan(x['hafe']) else x['hafe'], axis=1)
# dataset.to_pickle("trainset(224).pkl")
# dataset.drop(columns=['hafe'], inplace=True)

# %%
dataset.groupby('kfold').count()
# %%
kfold1 = dataset[dataset['kfold'].isin([1, 2])].copy()
kfold2 = dataset[dataset['kfold'].isin([1, 3])].copy()
kfold3 = dataset[dataset['kfold'].isin([2, 3])].copy()

# %%
kfold1.to_pickle("trainset_224_1of3.pkl")
kfold2.to_pickle("trainset_224_2of3.pkl")
kfold3.to_pickle("trainset_224_3of3.pkl")
