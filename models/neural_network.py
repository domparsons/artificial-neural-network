from typing import Any

import numpy as np
import random

import pandas as pd
from numpy import floating


class NeuralNetwork:
    """
    A class used to represent a Neural Network.

    Attributes
    ----------
    input_size : int
        The number of input features.
    hidden_layer_size : int
        The number of neurons in the hidden layer.
    output_size : int
        The number of output neurons.
    hidden_layer_weights : np.ndarray
        Weights of the hidden layer.
    hidden_layer_biases : np.ndarray
        Biases of the hidden layer.
    output_layer_weights : np.ndarray
        Weights of the output layer.
    output_layer_bias : float
        Bias of the output layer.
    previous_output_weight_change : np.ndarray
        Previous weight change for the output layer (for momentum).
    previous_hidden_weight_change : np.ndarray
        Previous weight change for the hidden layer (for momentum).

    Methods
    -------
    train(training_data, validation_data, epochs, initial_learning_rate, final_learning_rate, momentum_rate, epoch_split)
        Trains the neural network using the provided training and validation data.
    forward_pass(inputs)
        Performs a forward pass through the neural network.
    backward_pass(inputs, output_activations, hidden_layer_activations)
        Performs a backward pass through the neural network to calculate deltas.
    update_weights(inputs, hidden_activations, deltas, learning_rate, momentum_rate)
        Updates the weights of the neural network using the calculated deltas, learning rate, and momentum rate.
    calculate_mean_validation_error(validation_data)
        Calculates the mean validation error for the neural network.
    sigmoid(x)
        Applies the sigmoid activation function.
    simulated_annealing(start_lr, end_lr, max_epochs, current_epoch)
        Calculates the learning rate using simulated annealing.
    """
    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.hidden_layer_weights = np.random.randn(input_size, hidden_layer_size) * 0.01
        self.hidden_layer_biases = np.array([random.uniform(-0.1, 0.1) for _ in range(hidden_layer_size)])

        self.output_layer_weights = np.random.randn(hidden_layer_size, output_size)
        self.output_layer_bias = random.uniform(-0.1, 0.1)

        self.previous_output_weight_change = np.zeros_like(self.output_layer_weights)
        self.previous_hidden_weight_change = np.zeros_like(self.hidden_layer_weights)

    def train(self, training_data: pd.DataFrame, validation_data: pd.DataFrame, epochs: int,
              initial_learning_rate: float,
              final_learning_rate: float, momentum_rate: float, epoch_split: int) -> tuple:
        """
        Train the neural network using the provided training and validation data.

        Parameters:
        training_data (pd.DataFrame): The data used for training the neural network.
        validation_data (pd.DataFrame): The data used for validating the neural network.
        epochs (int): The number of epochs to train the neural network.
        initial_learning_rate (float): The initial learning rate for training.
        final_learning_rate (float): The final learning rate for training.
        momentum_rate (float): The momentum rate for weight updates.
        epoch_split (int): The interval at which to calculate and print the mean validation error.

        Returns:
        tuple: A tuple containing the mean validation errors, hidden layer weights, hidden layer biases, output layer weights, and output layer bias.
        """
        mean_validation_errors = []

        for epoch in range(epochs):
            learning_rate = self.simulated_annealing(initial_learning_rate, final_learning_rate, epochs, epoch)

            for _, row in training_data.iterrows():
                hidden_layer_activations, output_layer_activations = self.forward_pass(row.values)
                deltas = self.backward_pass(row.values, output_layer_activations, hidden_layer_activations)

                self.update_weights(row.values, hidden_layer_activations, deltas, learning_rate, momentum_rate)

            if epoch % epoch_split == 0:
                mean_validation_error = self.calculate_mean_validation_error(validation_data)
                print(f"{epoch}: Error: {mean_validation_error}")
                mean_validation_errors.append(mean_validation_error)

        return mean_validation_errors, self.hidden_layer_weights, self.hidden_layer_biases, self.output_layer_weights, self.output_layer_bias

    def forward_pass(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform a forward pass through the neural network.

        Parameters:
        inputs (np.ndarray): The input data.

        Returns:
        tuple: A tuple containing the hidden layer activations and output layer activations.
        """
        hidden_layer_weighted_sums = np.dot(inputs, self.hidden_layer_weights) + self.hidden_layer_biases
        hidden_layer_activations = self.sigmoid(hidden_layer_weighted_sums)

        output_layer_weighted_sums = np.dot(hidden_layer_activations, self.output_layer_weights) + self.output_layer_bias
        output_layer_activations = self.sigmoid(output_layer_weighted_sums)

        return hidden_layer_activations, output_layer_activations

    def backward_pass(self, inputs: np.ndarray, output_activations: np.ndarray,
                      hidden_layer_activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform a backward pass through the neural network to calculate deltas.

        Parameters:
        inputs (np.ndarray): The input data.
        output_activations (np.ndarray): The activations from the output layer.
        hidden_layer_activations (np.ndarray): The activations from the hidden layer.

        Returns:
        tuple: A tuple containing deltas for the output layer and hidden layer.
        """
        output_deltas = (inputs[-1] - output_activations[0]) * (output_activations[0] * (1 - output_activations[0]))
        hidden_deltas = (hidden_layer_activations * (1 - hidden_layer_activations)) * np.dot(output_deltas,
                                                                                             self.output_layer_weights.T)

        return output_deltas, hidden_deltas

    def update_weights(self, inputs: np.ndarray, hidden_activations: np.ndarray, deltas: list, learning_rate: float,
                       momentum_rate: float) -> None:
        """
        Update the weights of the neural network using the calculated deltas, learning rate, and momentum rate.

        Parameters:
        inputs (np.ndarray): The input data.
        hidden_activations (np.ndarray): The activations from the hidden layer.
        deltas (list): The calculated deltas for weight updates.
        learning_rate (float): The learning rate for weight updates.
        momentum_rate (float): The momentum rate for weight updates.
        """
        output_weight_change = learning_rate * np.outer(hidden_activations, deltas[0])
        hidden_weight_change = learning_rate * np.outer(inputs, deltas[1:])

        output_weight_change += momentum_rate * self.previous_output_weight_change
        hidden_weight_change += momentum_rate * self.previous_hidden_weight_change

        self.output_layer_weights += output_weight_change
        self.output_layer_bias += learning_rate * deltas[0]

        self.hidden_layer_weights += hidden_weight_change

        self.previous_output_weight_change = output_weight_change
        self.previous_hidden_weight_change = hidden_weight_change

    def calculate_mean_validation_error(self, validation_data: pd.DataFrame) -> floating[Any]:
        """
        Calculate the mean validation error for the neural network.

        Parameters:
        validation_data (pd.DataFrame): The data used for validating the neural network.

        Returns:
        float: The mean validation error.
        """
        validation_errors = []

        for _, val_row in validation_data.iterrows():
            _, val_output_layer_activations = self.forward_pass(val_row.values)
            error = np.mean((val_row.iloc[-1] - val_output_layer_activations[0]) ** 2) ** 0.5
            validation_errors.append(error)

        return np.mean(validation_errors)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def simulated_annealing(start_lr: float, end_lr: float, max_epochs: int, current_epoch: int) -> float:
        """
        Calculate the learning rate using simulated annealing.

        Parameters:
        start_lr (float): The initial learning rate.
        end_lr (float): The final learning rate.
        max_epochs (int): The total number of epochs.
        current_epoch (int): The current epoch number.

        Returns:
        float: The calculated learning rate for the current epoch.
        """
        new_rate = ((end_lr + (start_lr - end_lr)) * (1 - (1 / (1 + np.exp(10 - ((20 * current_epoch) / max_epochs))))))
        if new_rate > end_lr:
            return new_rate
        else:
            return end_lr
