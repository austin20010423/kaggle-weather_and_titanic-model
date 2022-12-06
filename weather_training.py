import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error


data_set = pd.read_csv('SimpleWeather.csv')
x = data_set.iloc[:, 2:8].values.reshape(-1, 6)
y = data_set.iloc[:, 1].values.reshape(-1, 1)

# linear training
train_data, test_data, train_target, test_target\
    = train_test_split(x, y, test_size=0.25, random_state=13)


LR = LinearRegression()
LR.fit(train_data, train_target)
print('the value of default measurement of linear regression:%.4f' %
      (LR.score(train_data, train_target)*100), '%')


# predicted result
train_pred = LR.predict(train_data)
test_pred = LR.predict(test_data)

print('train data temperature predict: \n', train_pred,
      '\n', 'test data temperature predict:\n', test_pred)


print('\nMSE train:%.3f,test:%.3f' % (mean_squared_error(train_target,
      train_pred), mean_squared_error(test_target, test_pred)))
