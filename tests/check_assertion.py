"""
Check input assertion
"""
# import library
from linear_regression_scratch.linear_model import LinearRegression

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np

# load and split data
diabetes_data = load_diabetes()
x = diabetes_data.data
y = diabetes_data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)

"""
Test assertion in LinearRegression parameters
"""
# define model

lr_model = LinearRegression(loss_function='mse')
# no error

# lr_model = LinearRegression(loss_function='abc')
# expected error = 'The loss abc is currently not supported.'

# lr_model = LinearRegression(n_iter=0)
# expected error = 'n_iter value cant be 0 or minus'

# lr_model = LinearRegression(learning_rate=-0.001)
# expected error = 'learning_rate value cant be 0 or minus'

# lr_model = LinearRegression(learning_rate=1)
# expected warnings = 'learning_rate value is not float'

"""
Test assertion in fit input length
"""
# fit and predict

# change x length
# x_train = x_train[:200]
# expected error = 'x and y have different length'

lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)

# The coefficients
print('Coefficients: ', lr_model.coef_)
print('Intercept: ', lr_model.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))