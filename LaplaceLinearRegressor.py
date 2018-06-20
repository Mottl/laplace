#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear regression for Laplace distributed targets (e.g. error = mean absolute error)
Author: Dmitry A. Mottl (https://github.com/Mottl)
License: MIT

Refer: https://en.wikipedia.org/wiki/Laplace_distribution
"""

import warnings
import numpy as np
from scipy import optimize

class LaplaceLinearRegressor():
    """Linear regression for Laplace distributed targets.
    Model minimizes mean absolute error (mae)
    instead of mean squared error (mse/rmse) for Gaussian distributed targets.
    """
    def __init__(self):
        self.w = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit linear model.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training data

        y : numpy array of shape [n_samples, 1]
            Target values.

        sample_weight : numpy array of shape [n_samples]
            Individual weights for each sample
            default is None — er
        """
        if not isinstance(X, np.ndarray):
            raise Exception("X must be of np.ndarray class")

        if len(X.shape) not in (1,2):
            raise Exception("X must be 2-dimensional array")

        if len(y.shape) != 1:
            raise Exception("y must be 1-dimensional array of targets")

        if len(X.shape) == 1:
            X_ = X.reshape(-1,1)
        else:
            X_ = X

        initial_weights = np.ones(X_.shape[1]+1)
        res = optimize.minimize(
            self.__mae, initial_weights,
            args=(X_, y, sample_weight),
            method='L-BFGS-B')

        if not res.success:
            warnings.warn(str(res), category=RuntimeWarning)
            raise Exception("scipy.optimize.minimize() didn't converge in LaplaceLinearRegressor.fit()")
        self.w = res.x

    def predict(self, X):
        if self.w is None:
            raise Exception(("This LaplaceLinearRegressor instance is not fitted yet."
                "Call 'fit' with appropriate arguments before using this method."))
        return w[0] + np.matmul(x, w[1:])

    def get_params():
        raise("Not implemented")


    def set_params():
        raise("Not implemented")


    def __mae(self, w, x, y, sample_weight):
        """
        w - vector params of linear regression of shape (D+1, 1)
        x - matrix NxD
        y - vector of targets (Nx1)
        """
        y_pred = w[0] + np.matmul(x, w[1:])

        if sample_weight is None:
            return np.sum(np.fabs(y - y_pred)) / y.shape[0]
        else:
            return np.sum(np.dot(np.fabs(y - y_pred), sample_weight)) / y.shape[0]

if __name__ == '__main__':
    x = np.linspace(0,10,20)
    y = 2*x + 5
    y[0] = 20
    y[1] = 25
    y[-1] = 10
    y[-2] = 5

    llr = LaplaceLinearRegressor()
    llr.fit(x, y)
    print("weights =",llr.w)
