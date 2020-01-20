# %%
import pandas as pd
import numpy as np
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
        if '.' in w:
            title[i] = w
            break

unique_title = set(title)

for t in unique_title:
    print(t, title.count(t))

# Use the Miss titles to divide female into to groups
# Other specical titles gives two infomations
#   nob. or not
#   service or not
# %% check Cabin
for i in range(1, 4):
    temp = raw_data.train['Cabin'][raw_data.train['Pclass'] == i]
    print(temp.unique())

# in any of the classes there are missing data.
# It looks like 1st class has less missin data.
# I just wonder why the data are missing?
# Is it because that dead people cannot report their cabin?
# I tend to no use this information
# If I have to, I will first try a simple lable: missing or not missing.
# %% pre processing data
def convert_data(data_frame):
    data = pd.DataFrame()
    # Pclass
    classes = data_frame['Pclass'].unique()
    classes.sort()
    for c in classes:
        data['Pc' + str(c)] = (1 * (data_frame['Pclass'] == c))

    # Special title
    ut = list(unique_title)
    ut.sort()
    # for i in range(len(ut)):
    #     print(i,ut[i])
    si = [0, 16, 1, 7]
    ei = [2, 8, 3, 6, 4, 15, 5, 11]
    service_title = {ut[i] for i in si}
    elite_title = {ut[i] for i in ei}
    data['Service'] = pd.Series([(1 if t in service_title else 0) for t in title])
    data['Elite'] = pd.Series([(1 if t in elite_title else 0) for t in title])

    # Gender
    data['Male'] = (1 * (data_frame['Sex'] == 'male'))
    male = data['Male'].tolist()
    unmarried = [t in {'Miss.', 'Mlle.'} for t in title]
    data['Female1'] = pd.Series([1 if male[i]==0 and unmarried[i] else 0 for i in range(len(male))])
    data['Female2'] = pd.Series([1 if male[i]==0 and not unmarried[i] else 0 for i in range(len(male))])

    # Age need impute
    data['Age'] = data_frame['Age'] 
    data['Age'] = data['Age'].fillna(data['Age'].median())

    # copy others
    data['SibSp'] = data_frame['SibSp']
    data['Parch'] = data_frame['Parch']
    data['Fare'] = data_frame['Fare']
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())

    # Embarked
    data['Embarked'] = (1 * (data_frame['Embarked'] == 'S'))
    return data


# visualize
data = convert_data(raw_data.train)
print(data.describe())
data.head()

# %% np data prepare
# training data
u = data.values
# train output
t = raw_data.train['Survived'].values
# test data
x = convert_data(raw_data.test).values
# %% sklearn methods test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(u, t, test_size=0.33)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

classifiers = [
    LogisticRegression(),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

y_vote = np.zeros_like(y_test) * 0.1
w = np.zeros(len(classifiers))
for i, clf in enumerate(classifiers):
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    w[i] = np.mean([1.0 if y_predict[i] == y_test[i] else 0.0 for i in range(len(y_test))])
    print(w[i])
    y_vote += w[i]**2 * y_predict

print('final-----------------------')
y_final = 1.0 * (y_vote > (np.sum(w**2) / 2))
print(np.mean([1 if y_final[i] == y_test[i] else 0 for i in range(len(y_test))]))


# %% out
y_vote = np.zeros(len(x)) * 0.1
for i, clf in enumerate(classifiers):
    clf.fit(u, t)
    y_predict = clf.predict(x)
    y_vote += w[i]**2 * y_predict
y_final = 1 * (y_vote > (np.sum(w**2) / 2))
out = raw_data.gender_submission.copy()
out['Survived'] = pd.Series(y_final)
out.to_csv('LR.csv', index=False)
# %%
