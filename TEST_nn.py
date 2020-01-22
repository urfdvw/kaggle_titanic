# %%
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import torch
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
    './titanic.zip')
raw_data.train.head()
# %% pre processing


class pre_processing:
    def __init__(self, data_frame):
        # get title
        names = data_frame['Name'].tolist()
        self.title = ['' for i in range(len(names))]
        for i, n in enumerate(names):
            words = n.split(' ')
            for w in words:
                if '.' in w:
                    self.title[i] = w
                    break
        # special titles
        unique_title = set(self.title)
        ut = list(unique_title)
        ut.sort()
        si = [0, 16, 1, 7]
        ei = [2, 8, 3, 6, 4, 15, 5, 11]
        self.service_title = {ut[i] for i in si}
        self.elite_title = {ut[i] for i in ei}

        # normalization
        self.mu_age = data_frame['Age'].mean()
        self.std_age = data_frame['Age'].std()
        self.mu_fare = data_frame['Fare'].mean()
        self.std_fare = data_frame['Fare'].std()
        return

    def __call__(self, data_frame):
        data = pd.DataFrame()
        # Pclass
        classes = data_frame['Pclass'].unique()
        classes.sort()
        for c in classes:
            data['Pc' + str(c)] = (1 * (data_frame['Pclass'] == c))

        # Special title
        data['Service'] = pd.Series(
            [(1 if t in self.service_title else 0) for t in self.title])
        data['Elite'] = pd.Series(
            [(1 if t in self.elite_title else 0) for t in self.title])

        # Gender
        data['Male'] = (1 * (data_frame['Sex'] == 'male'))
        male = data['Male'].tolist()
        unmarried = [t in {'Miss.', 'Mlle.'} for t in self.title]
        data['Female1'] = pd.Series(
            [1 if male[i] == 0 and unmarried[i] else 0 for i in range(len(male))])
        data['Female2'] = pd.Series(
            [1 if male[i] == 0 and not unmarried[i] else 0 for i in range(len(male))])

        # Age need impute
        data['Age'] = data_frame['Age']
        data['Age'] = data['Age'].fillna(data['Age'].median())
        data['Age'] = (data['Age'] - self.mu_age) / self.std_age

        # copy others
        data['SibSp'] = data_frame['SibSp']
        data['Parch'] = data_frame['Parch']
        data['Fare'] = data_frame['Fare']
        data['Fare'] = data['Fare'].fillna(data['Fare'].median())
        data['Fare'] = (data['Fare'] - self.mu_fare) / self.std_fare

        # Embarked
        data['Embarked'] = (1 * (data_frame['Embarked'] == 'S'))
        return data


pre_pro = pre_processing(raw_data.train)
data = pre_pro(raw_data.train)
data.head()

# %% df to np data prepare
# training data
u = data.values
# training output
t = raw_data.train['Survived'].values
# test data
x = pre_pro(raw_data.test).values


# %% sklearn methods test
x_train, x_test, y_train, y_test = train_test_split(u, t, test_size=0.33)


device = 'cuda'
net = nn.Sequential(
    nn.Linear(13, 26),
    nn.Tanhshrink(),
    nn.Linear(26, 52),
    nn.Tanhshrink(),
    nn.Linear(52, 26),
    nn.Tanhshrink(),
    nn.Linear(26, 13),
    nn.Tanhshrink(),
    nn.Linear(13, 6),
    nn.Tanhshrink(),
    nn.Linear(6, 3),
    nn.Tanhshrink(),
    nn.Linear(3, 1),
    nn.Sigmoid()
).to(device)

optimizer = optim.Adam(net.parameters())

T = 10000  # number of epoch
m = 100  # batch size
loss = nn.BCELoss().to(device)
for i in range(T):
    ind = np.random.choice(len(y_train), size=m, replace=False)
    x_cur, t_cur = torch.as_tensor(x_train[ind], dtype=torch.float).to(device), torch.as_tensor(
        y_train[ind], dtype=torch.float).to(device)
    l = loss(net(x_cur), t_cur)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()

t_test = (net(torch.as_tensor(x_test, dtype=torch.float).to(device)
              ).cpu().detach().numpy() > 0.5) * 1.0

print(np.mean([1 if t_test[i] == y_test[i]
               else 0 for i in range(len(y_test))]))


# %% out
T = 20000  # number of epoch
m = 100  # batch size
loss = nn.BCELoss().to(device)
for i in range(T):
    ind = np.random.choice(len(t), size=m, replace=False)
    x_cur, t_cur = torch.as_tensor(u[ind], dtype=torch.float).to(device), torch.as_tensor(
        t[ind], dtype=torch.float).to(device)
    l = loss(net(x_cur), t_cur)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
#%%
t_test = (net(torch.as_tensor(x, dtype=torch.float).to(device)
              ).cpu().detach().numpy() > 0.5) * 1
out = raw_data.gender_submission.copy()
out['Survived'] = pd.Series(t_test.flatten())
out.to_csv('LR.csv', index=False)
# %%
