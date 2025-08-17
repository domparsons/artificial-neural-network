import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error


class FloodPredictor:
    """
    FloodPredictor class for predicting flood index values using a trained neural network.

    Attributes:
        hidden_layer_weights (np.ndarray): Weights of the hidden layer.
        hidden_layer_biases (np.ndarray): Biases of the hidden layer.
        output_layer_weights (np.ndarray): Weights of the output layer.
        output_layer_bias (np.ndarray): Bias of the output layer.
    """

    def __init__(
        self,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_bias,
    ):
        self.hidden_layer_weights = hidden_layer_weights
        self.hidden_layer_biases = hidden_layer_biases
        self.output_layer_weights = output_layer_weights
        self.output_layer_bias = output_layer_bias

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def unstandardise_data(
        s: float, min_val: float, max_val: float, scaling_factor: float = 0.8
    ) -> float:
        """ "
        Convert standardised data back to its original scale.

        Parameters:
        s (float): The standardised value.
        min_val (float): The minimum value of the original data.
        max_val (float): The maximum value of the original data.
        scaling_factor (float): The scaling factor used during standardisation. Default is 0.8.

        Returns:
        float: The unstandardised value.
        """
        r = ((s - 0.1) / scaling_factor) * (max_val - min_val) + min_val
        return r

    def predict(
        self,
        standardised_testing_data: pd.DataFrame,
        y_test: pd.Series,
    ) -> list:
        """
        Predict the index flood values for the given standardised testing data.

        Parameters:
        standardised_testing_data (pandas.DataFrame): The standardised testing data.
        y_train (pandas.Series): Used only to unstandardise predictions.

        Returns:
        list: The predicted index flood values.
        """
        min_index_flood = y_test.min()
        max_index_flood = y_test.max()

        predictions = []
        for idx, row in standardised_testing_data.iterrows():
            hidden_layer_weighted_sums = (
                np.dot(row.values, self.hidden_layer_weights)
                + self.hidden_layer_biases
            )
            hidden_layer_activations = self.leaky_relu(
                hidden_layer_weighted_sums
            )

            output_layer_weighted_sums = (
                np.dot(hidden_layer_activations, self.output_layer_weights)
                + self.output_layer_bias
            )
            output_layer_activations = output_layer_weighted_sums
            pred = self.unstandardise_data(
                output_layer_activations[0], min_index_flood, max_index_flood
            )
            predictions.append(max(0, int(pred)))

        return predictions

    @staticmethod
    def evaluate(
        y_test: np.ndarray, y_predict: list, threshold: int = 60
    ) -> tuple:
        """
        Evaluate the performance of the flood prediction model.

        Parameters:
        y_test (array-like): The true index flood values.
        y_predict (array-like): The predicted index flood values.
        threshold (int, optional): The threshold value for precision calculation. Default is 60.

        Returns:
        tuple: A tuple containing the correlation coefficient, precision, and mean absolute error (MAE).
        """
        correlation_coefficient = pearsonr(y_test, y_predict)
        within_threshold = np.sum(np.abs(y_test - y_predict) < threshold)
        precision = within_threshold / len(y_test)
        mae = mean_absolute_error(y_test, y_predict)
        return correlation_coefficient, precision, mae
