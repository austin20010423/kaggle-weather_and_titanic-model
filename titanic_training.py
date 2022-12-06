import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# grab data from training
train['Title'] = train['Name'].str.split(',', expand=True)[1]
train['Title'] = train['Title'].str.split('.', expand=True)[0]
mean_age_train = train[['Title', 'Age']].groupby('Title').mean()


# grab data from testing
test['Title'] = test['Name'].str.split(',', expand=True)[1]
test['Title'] = test['Title'].str.split('.', expand=True)[0]
mean_age_test = test[['Title', 'Age']].groupby('Title').mean()


# add missing age data on training data
mapping_train = dict(zip(mean_age_train.index, mean_age_train['Age']))
for i in range(len(train['Age'].isnull())):
    if train['Age'].isnull()[i]:
        train.loc[i, 'Age'] = mapping_train.get(train['Title'][i])


# add missing data on testing data
mapping_test = dict(zip(mean_age_test.index, mean_age_test['Age']))
for i in range(len(test['Age'].isnull())):
    if test['Age'].isnull()[i]:
        test.loc[i, 'Age'] = mapping_test.get(test['Title'][i])


# drop no need data
train.drop('Name', inplace=True, axis=1)
train.drop('Cabin', inplace=True, axis=1)
train.drop('Ticket', inplace=True, axis=1)

test.drop(414, inplace=True, axis=0)
test.drop('Name', inplace=True, axis=1)
test.drop('Cabin', inplace=True, axis=1)
test.drop('Ticket', inplace=True, axis=1)
test = test.fillna(28)

# print(test.isnull().sum())
# encoding train data
sexual = train.values[:, 3].reshape(-1, 1)
embarked = train.values[:, 8].reshape(-1, 1)
title = train.values[:, 9].reshape(-1, 1)

sexual_test = test.values[:, 2].reshape(-1, 1)
embarked_test = test.values[:, 7].reshape(-1, 1)
title_test = test.values[:, 8].reshape(-1, 1)


encode = OneHotEncoder()
x = encode.fit_transform(sexual).toarray()
y = encode.fit_transform(embarked).toarray()
z = encode.fit_transform(title).toarray()

x_t = encode.fit(sexual).transform(sexual_test).toarray()
y_t = encode.fit(embarked).transform(embarked_test).toarray()
z_t = encode.fit(title).transform(title_test).toarray()


sexual = np.hstack([train.values[:, 1].reshape(-1, 1), train.values[:, 2].reshape(-1, 1),
                    x, train.values[:, 4].reshape(-1,
                                                  1), train.values[:, 5].reshape(-1, 1),
                    train.values[:, 6].reshape(-1,
                                               1), train.values[:, 7].reshape(-1, 1),
                    y, z])

sexual_test = np.hstack([test.values[:, 1].reshape(-1, 1), x_t, test.values[:, 3].reshape(-1, 1), test.values[:,
                        4].reshape(-1, 1), test.values[:, 5].reshape(-1, 1), test.values[:, 6].reshape(-1, 1), y_t, z_t])


train = pd.DataFrame(sexual)
test = pd.DataFrame(sexual_test)
train = train.astype('float')
test = test.astype('float')

# train
data = train.iloc[:, 1:29].values.reshape(-1, 28)
target = train.iloc[:, 0].values.reshape(-1, 1)

train_data, test_data, train_target, test_target\
    = train_test_split(data, target, test_size=0.25, random_state=13)

LR = LogisticRegression()
LR.fit(train_data, train_target.flatten())

# predict
train_pre = LR.predict(train_data)
train_pre = pd.DataFrame(train_pre)
test_pre = LR.predict(test_data)

print('model score')
print('the train data set score is :%d' % (LR.score(train_data, train_pre)))
print('train data set performance :%.3f' % (log_loss(train_target, train_pre)))
print('the test data set score is :%d' % (LR.score(test_data, test_pre)))
print('test data set performance :%.3f' % (log_loss(test_target, test_pre)))

# use model to predict test.csv data
data_test = test.iloc[:, 0:28].values.reshape(-1, 28)
predict_test_data = LR.predict(data_test)


# check the correctness of test data
gender_submission = pd.read_csv('gender_submission.csv')
gender_submission.drop(414, inplace=True, axis=0)
gender_submission.drop('PassengerId', inplace=True, axis=1)
gender_submission = gender_submission.astype('float')

gender_submission = gender_submission.iloc[:, 0].values.reshape(-1, 1)
print('test data to gender submission performance :%.3f' %
      (log_loss(gender_submission, predict_test_data)))
