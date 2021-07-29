"""
Linear Regression Model From Scratch
"""
import numpy as np
import warnings

class LinearRegression:
    """
    LINEAR REGRESSION MODEL

    PARAMETERS
    ----------
    loss_function: str = 'mse'
        The loss function to be used. Defaults to 'mse', which gives a mean
        squared error. The possible options are 'mse' = mean squared error',
        mae' = mean absolute error, and 'rmse' = root mean squared error.
    n_iter: int = 10000
        The amount of iteration.
    learning_rate: float = 0.01
        Learning rate value.
    """
    def __init__(
                self, 
                loss_function: str = 'mse',
                n_iter: int = 10000, 
                learning_rate: float = 0.01
                ):

        self.n_iter = self._assert_niter(n_iter)
        self.learning_rate = self._assert_lr(learning_rate)
        self.loss = []
        self.coef_, self.intercept_ = None, None

        if loss_function == 'mse':
            self.loss_function = self._mean_squared_error
        elif loss_function == 'mae':
            self.loss_function = self._mean_absolute_error
        elif loss_function == 'rmse':
            self.loss_function = self._root_mean_squared_error
        else:
            raise ValueError("The loss %s is currently not supported. " % loss_function)

    @staticmethod
    def _assert_niter(value):
        try:

            assert value > 0

        except AssertionError as exc:

            raise AssertionError(

                "n_iter value cant be 0 or minus"
                
            ) from exc
        
        return value

    @staticmethod
    def _assert_lr(value):
        try:

            assert value > 0

        except AssertionError as exc:

            raise AssertionError(

                "learning_rate cant be 0 or minus"
                
            ) from exc
        
        if type(value) != float:
            warnings.warn("learning_rate value is not float")
            return float(value)
        else:
            return value

    @staticmethod
    def _mean_squared_error(y_true, y_pred):
        """
        Get mean squared error of prediction

        PARAMETERS
        ----------
        y_true: np.ndarray
            true values
        y_pred: np.ndarray
            predicted values

        RETURNS
        -------
        float
            mean squared error value from the prediction
        """
        error = 0
        for i in range(len(y_true)):
            error += (y_true[i] - y_pred[i]) ** 2
        return error / len(y_true)
    
    @staticmethod
    def _mean_absolute_error(y_true, y_pred):
        """
        Get mean absolute error of prediction

        PARAMETERS
        ----------
        y_true: np.ndarray
            true values
        y_pred: np.ndarray
            predicted values

        RETURNS
        -------
        float
            mean absolute error value from the prediction
        """
        error = 0
        for i in range(len(y_true)):
            error += abs(y_true[i] - y_pred[i])
        return error / len(y_true)

    @staticmethod
    def _root_mean_squared_error(y_true, y_pred):
        """
        Get root mean absolute error of prediction

        PARAMETERS
        ----------
        y_true: np.ndarray
            true values
        y_pred: np.ndarray
            predicted values

        RETURNS
        -------
        float
            root mean absolute error value from the prediction
        """
        error = 0
        for i in range(len(y_true)):
            error += (y_true[i] - y_pred[i]) ** 2
        return np.sqrt(error / len(y_true))

    def fit(self, x, y):
        """
        Fitting models to data

        PARAMETERS
        ----------
        x: np.ndarray
            features
        y: np.ndarray
            true values
        """
        # handle the difference in length of x and y
        try:

            assert len(x) == len(y)

        except AssertionError as exc:

            raise AssertionError(

                "x and y have different length"
                
            ) from exc

        # initialize coef_ and intercept_
        self.coef_ = np.zeros(x.shape[1])
        self.intercept_ = 0
        
        # gradient descent
        for i in range(self.n_iter):
            # calculate y_pred and get loss value
            y_pred = np.dot(x, self.coef_) + self.intercept_
            loss = self.loss_function(y, y_pred)
            self.loss.append(loss)
            
            # calculate derivatives
            D_c = (-2 / x.shape[0]) * (np.dot(x.T, (y-y_pred)))
            D_i = (-2 / x.shape[0]) * (np.sum(y-y_pred))
            
            # update coefficients and intercept
            self.coef_ -= self.learning_rate * D_c
            self.intercept_ -= self.learning_rate * D_i
        
        
    def predict(self, x):
        """
        Make predictions

        PARAMETERS
        ----------
        x: np.ndarray
            features
        RETURNS
        -------
        np.ndarray
            predictions
        """
        return np.dot(x, self.coef_) + self.intercept_