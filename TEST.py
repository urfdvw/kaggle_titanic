#%%
import pandas as pd
import zipfile

"""
class that read zipped csv files as an onject.
"""
class data_reader:
    def __init__(self, path):
        zf = zipfile.ZipFile(path)
        for fn in zipfile.ZipFile.namelist(zf):
            vn = fn.split('.')[0] # remove '.csv'
            setattr(self, vn, pd.read_csv(zf.open(fn)))
        return

#%%
data = data_reader('/home/hwang/Documents/Github/kaggle_titanic/titanic.zip')
data.train.head()

# %%
