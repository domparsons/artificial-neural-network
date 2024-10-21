import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

class FloodPredictor:

    def __init__(self, hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias):
        self.hidden_layer_weights = hidden_layer_weights
        self.hidden_layer_biases = hidden_layer_biases
        self.output_layer_weights = output_layer_weights
        self.output_layer_bias = output_layer_bias

    # Activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Unstandardising function
    @staticmethod
    def unstandardise_data(s, min_val, max_val, scaling_factor=0.8):
        r = ((s - 0.1) / scaling_factor) * (max_val - min_val) + min_val
        return r

    # A function to predict 'Index flood' values for standardised testing data
    def predict(self, standardised_testing_data, train_and_validate):
        # Calculate min and max values for 'Index flood'
        min_index_flood = train_and_validate['Index flood'].min()
        max_index_flood = train_and_validate['Index flood'].max()

        # Perform forward pass to get activations at hidden and output layers for the testing data
        hidden_layer_activations = self.sigmoid(np.dot(standardised_testing_data.iloc[:, :-1], self.hidden_layer_weights[:-1, :]) + self.hidden_layer_biases)
        output_layer_activations = self.sigmoid(np.dot(hidden_layer_activations, self.output_layer_weights) + self.output_layer_bias)

        # Unstandardise the output using training set min and max values for 'Index flood'
        predictions = [self.unstandardise_data(output[0], min_index_flood, max_index_flood) for output in output_layer_activations]

        # Ensure predictions are non-negative
        predictions = [max(0, pred) for pred in predictions]

        return predictions

    @staticmethod
    def evaluate(y_test, y_predict, threshold=60):
        # Calculate the Pearson correlation coefficient between actual and predicted values
        correlation_coefficient, _ = pearsonr(y_test, y_predict)
        # Calculate precision as the ratio of predicted values within the threshold to the total number of test samples
        within_threshold = np.sum(np.abs(y_test - y_predict) < threshold)
        precision = within_threshold / len(y_test)
        # Calculate mean absolute error
        mae = mean_absolute_error(y_test, y_predict)
        return correlation_coefficient, precision, mae