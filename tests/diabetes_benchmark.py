"""
Benchmark the result of LinearRegression from linear_regression_scratch and sklearn
"""
# import library
from linear_regression_scratch.linear_model import LinearRegression

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import linear_model

# load and split
diabetes_data = load_diabetes()
x = diabetes_data.data
y = diabetes_data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2021)

# define, fit and predict (linear_regression_scratch)
lr_model = LinearRegression(loss_function='mse', learning_rate=0.5)
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)

# define, fit and predict (sklearn)
sklearn_lr_model = linear_model.LinearRegression()
sklearn_lr_model.fit(x_train, y_train)
sklearn_y_pred = sklearn_lr_model.predict(x_test)

# linear_regression_scratch
print('Result from linear_regression_scratch')
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, y_pred))

# linear_regression_scratch
print('Result from sklearn')
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, sklearn_y_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, sklearn_y_pred))