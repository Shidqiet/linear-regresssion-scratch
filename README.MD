# Linear Regression Model From Scratch (Numpy Only)
Online Test Research Scientist Application for Pacmann AI

By: Shidqie Taufiqurrahman

## Requirements
1. numpy
2. sklearn (for load the dataset and evaluate prediction)

Install requirements:
```bash
pip install -r requirements.txt
```

## Packgage installation
```bash
python setup.py build
python setup.py install
```

## Package usage
```python
from linear_regression_scratch.linear_model import LinearRegression
lr_model = LinearRegression(loss_function='mse', learning_rate=0.5)
lr_model.fit(x_train, y_train)
y_pred = lr_model.predict(x_test)
```

### LinearRegression parameters
* loss_function: str, default = 'mse'

    The loss function to be used. Defaults to 'mse', which gives a mean squared error. The possible options are:
    * 'mse' = mean squared error
    * 'mae' = mean absolute error
    * 'rmse' = root mean squared error
    
* n_iter: int, default = 10000

    The amount of iteration. Cant be 0 or minus

* learning_rate: float, default = 0.01

    Learning rate value. Cant be 0 or minus

## Demo/Tests
1. Library test

Make sure the package/library are installed correctly
```bash
python tests/lib_test.py
```
2. Demo on sklearn diabetes dataa
```bash
python tests/diabetes_demo.py
```
3. Benchmark the result of LinearRegression from linear_regression_scratch and sklearn on diabetes data
```bash
python tests/diabetes_benchmark.py
```