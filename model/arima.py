import tensorflow as tf
import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


# Class for the linear regression model
class Arima:
    def __init__(self, order=(1,1,0)):
        self.order = order
        self.model = SARIMAX([])
        self.model_fit = None

    def compile(self):
        # Compile the model with the optimizer and the loss function for regression
        self.model.compile(metrics=["cosine_similarity", tf.keras.metrics.MeanAbsoluteError()])

    def get_weights(self, model_fit):
        if model_fit is None:
            return []
        return model_fit.params

    def set_weights(self, parameters):
        if self.model_fit is None:
            return
        self.model_fit.params = parameters

    def bind(self, target, exog_features):
        """
        Rebinds new data to an existing SARIMAX model by reinitializing it with the same configuration.

        Args:
            existing_model (SARIMAXResults): The existing fitted SARIMAX model.
            new_data (list or ndarray): The new time series data.

        Returns:
            SARIMAXResults: A fitted SARIMAX model with the new data.
        """

        # Initialize a new SARIMAX model with the same configuration
        model_new = SARIMAX(endog=target, exog=exog_features, order=self.order)

        # Fit the new model (new data) starting with the older params and config
        if self.model_fit is not None:
            fitted_model = model_new.fit(start_params=self.model_fit.params, disp=False)
        else:
            fitted_model = model_new.fit(disp=False)

        self.model_fit = fitted_model
        return fitted_model

    def get_model(self):
        return self.model
