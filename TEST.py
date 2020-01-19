# %%
import pandas as pd
import zipfile
import matplotlib.pyplot as plt
# %% read data as pd


class data_reader:
    """
    class that read zipped csv files as an onject.
    """

    def __init__(self, path):
        zf = zipfile.ZipFile(path)
        for fn in zipfile.ZipFile.namelist(zf):
            vn = fn.split('.')[0]  # remove '.csv'
            setattr(self, vn, pd.read_csv(zf.open(fn)))
        return


raw_data = data_reader(
    '/home/hwang/Documents/Github/kaggle_titanic/titanic.zip')
raw_data.train.head()
# %% simple guesses and visualization #############################

# PassengerId is only used for output as it is just index

# Survived is the output

# %% Pclass might not be very useful because the information is more detailed in Fare
# the following code is used to test this idea

for c in raw_data.test['Pclass'].unique():
    temp = raw_data.train['Fare'][raw_data.train['Pclass'] == c]
    plt.figure()
    temp.hist(bins=50)
    plt.show()
    print('class', c)
    print(temp.describe())

# No, the price is not the same for any class

# %% are the babies free
temp = raw_data.train['Fare'][raw_data.train['Age'] <= 2.0]
plt.figure()
temp.hist(bins=50)
plt.show()
print('babies')
print(temp.describe())

# %% Name, see how many titles are there

names = raw_data.train['Name'].tolist()
title = ['' for i in range(len(names))]
for i, n in enumerate(names):
    words = n.split(' ')
    for w in words:
        if '.' in w :
            title[i] = w
            break

unique_title = set(title)

for t in unique_title:
    print(t, title.count(t))

# Use the Miss titles to divide female into to groups
# Other specical titles gives two infomations
#   nob. or not
#   service or not

# TODO

# %% check Cabin
for i in range(1, 4):
    temp = raw_data.train['Cabin'][raw_data.train['Pclass'] == i]
    print(temp.unique())

# in any of the classes there are missing data.
# It looks like 1st class has less missin data.
# I just wonder why the data are missing? 
# Is is because that dead people cannot report their cabin?
# I tend to no use this information
# If I have to, I will first try a simple lable: missing or not missing.

# %%
