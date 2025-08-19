from typing import Any

import numpy as np
import random

import pandas as pd
from numpy import floating


class NeuralNetwork:
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        limit = np.sqrt(6 / (input_size + hidden_layer_size))

        self.hidden_layer_weights = np.random.uniform(
            -limit, limit, size=(input_size, hidden_layer_size)
        )

        self.hidden_layer_biases = np.array(
            [random.uniform(-0.1, 0.1) for _ in range(hidden_layer_size)]
        )

        self.output_layer_weights = np.random.uniform(
            -limit, limit, size=(hidden_layer_size, output_size)
        )

        self.output_layer_bias = np.zeros(self.output_size)

        self.previous_output_weight_change = np.zeros_like(
            self.output_layer_weights
        )
        self.previous_hidden_weight_change = np.zeros_like(
            self.hidden_layer_weights
        )

    def train(
        self,
        x_train_standardised: pd.DataFrame,
        x_val_standardised: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        epochs: int,
        initial_learning_rate: float,
        final_learning_rate: float,
        momentum_rate: float,
        epoch_split: int = 1,
    ) -> tuple:
        mean_validation_errors = []

        import time

        start_time = time.time()

        X = x_train_standardised.to_numpy(dtype=float)
        y = y_train.to_numpy(dtype=float).reshape(-1, 1)

        for epoch in range(epochs):
            learning_rate = self.simulated_annealing(
                initial_learning_rate, final_learning_rate, epochs, epoch
            )

            (
                hidden_layer_activations,
                output_layer_activations,
                hidden_layer_weighted_sums,
            ) = self.forward_pass(X)

            output_delta, hidden_delta = self.backward_pass(
                y,
                output_layer_activations,
                hidden_layer_weighted_sums,
            )
            self.update_weights(
                X,
                hidden_layer_activations,
                output_delta,
                hidden_delta,
                learning_rate,
                momentum_rate,
            )

            if epoch % epoch_split == 0:
                mean_validation_error = self.calculate_mean_validation_error(
                    x_val_standardised, y_val
                )
                print(f"{epoch}: Error: {mean_validation_error}")
                mean_validation_errors.append(mean_validation_error)

        end_time = time.time()

        print(end_time - start_time)

        return (
            mean_validation_errors,
            self.hidden_layer_weights,
            self.hidden_layer_biases,
            self.output_layer_weights,
            self.output_layer_bias,
        )

    def forward_pass(
        self, inputs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        hidden_layer_weighted_sums = (
            np.dot(inputs, self.hidden_layer_weights) + self.hidden_layer_biases
        )
        hidden_layer_activations = self.leaky_relu(hidden_layer_weighted_sums)

        output_layer_weighted_sums = (
            np.dot(hidden_layer_activations, self.output_layer_weights)
            + self.output_layer_bias
        )
        output_layer_activations = output_layer_weighted_sums

        return (
            hidden_layer_activations,
            output_layer_activations,
            hidden_layer_weighted_sums,
        )

    def backward_pass(
        self,
        y_true: np.ndarray,
        output_activations: np.ndarray,
        hidden_layer_weighted_sums: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        y_pred = output_activations
        output_delta = y_true - y_pred

        hidden_delta = self.leaky_relu_derivative(
            hidden_layer_weighted_sums
        ) * np.dot(output_delta, self.output_layer_weights.T)

        return output_delta, hidden_delta

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        return np.where(x > 0, 1.0, alpha)

    def update_weights(
        self,
        inputs: np.ndarray,
        hidden_activations: np.ndarray,
        output_delta: np.ndarray,
        hidden_delta: np.ndarray,
        learning_rate: float,
        momentum_rate: float,
    ) -> None:
        batch_size = inputs.shape[0]

        grad_output_weights = (
            np.dot(hidden_activations.T, output_delta) / batch_size
        )
        grad_hidden_weights = np.dot(inputs.T, hidden_delta) / batch_size

        grad_output_bias = np.mean(output_delta, axis=0)
        grad_hidden_bias = np.mean(hidden_delta, axis=0)

        output_weight_change = (
            learning_rate * grad_output_weights
            + momentum_rate * self.previous_output_weight_change
        )
        hidden_weight_change = (
            learning_rate * grad_hidden_weights
            + momentum_rate * self.previous_hidden_weight_change
        )

        self.output_layer_weights += output_weight_change
        self.output_layer_bias += learning_rate * grad_output_bias
        self.hidden_layer_weights += hidden_weight_change
        self.hidden_layer_biases += learning_rate * grad_hidden_bias

        self.previous_output_weight_change = output_weight_change
        self.previous_hidden_weight_change = hidden_weight_change

    def calculate_mean_validation_error(
        self, validation_data: pd.DataFrame, y_val: pd.Series
    ) -> floating[Any]:
        validation_errors = []

        for idx, val_row in validation_data.iterrows():
            _, val_output_layer_activations, _ = self.forward_pass(
                val_row.values
            )
            y_true = y_val.loc[idx]
            y_pred = val_output_layer_activations[0]

            error = np.sqrt((y_true - y_pred) ** 2)
            validation_errors.append(error)

        return np.mean(validation_errors)

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def simulated_annealing(
        start_lr: float, end_lr: float, max_epochs: int, current_epoch: int
    ) -> float:
        new_rate = (end_lr + (start_lr - end_lr)) * (
            1 - (1 / (1 + np.exp(10 - ((20 * current_epoch) / max_epochs))))
        )
        if new_rate > end_lr:
            return new_rate
        else:
            return end_lr
