import numpy as np
import random

class NeuralNetwork:

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

    def train(self, training_data, validation_data, epochs, initial_learning_rate, final_learning_rate, momentum_rate,
              epoch_split):
        mean_validation_errors = []

        for epoch in range(epochs):
            # Update the learning rate
            learning_rate = self.simulated_annealing(initial_learning_rate, final_learning_rate, epochs, epoch)

            # Iterate through the training data rows
            for _, row in training_data.iterrows():
                # Forward pass
                hidden_layer_activations, output_layer_activations = self.forward_pass(row.values)

                # Backward pass
                deltas = self.backward_pass(row.values, output_layer_activations, hidden_layer_activations)

                # Update weights and biases
                self.update_weights(row.values, hidden_layer_activations, deltas, learning_rate, momentum_rate)

            if epoch % epoch_split == 0:
                # Calculate mean validation error
                mean_validation_error = self.calculate_mean_validation_error(validation_data)

                # Print the mean validation error for the current epoch
                print(f"{epoch}: Error: {mean_validation_error}")

                # Append the mean validation error to the list of mean validation errors
                mean_validation_errors.append(mean_validation_error)

        return mean_validation_errors, self.hidden_layer_weights, self.hidden_layer_biases, self.output_layer_weights, self.output_layer_bias

    def forward_pass(self, inputs):
        # Calculate weighted sums at the hidden layer
        hidden_layer_weighted_sums = np.dot(inputs, self.hidden_layer_weights) + self.hidden_layer_biases
        # Apply sigmoid activation to get activations at the hidden layer
        hidden_layer_activations = self.sigmoid(hidden_layer_weighted_sums)

        # Calculate weighted sums at the output layer
        output_layer_weighted_sums = np.dot(hidden_layer_activations,
                                            self.output_layer_weights) + self.output_layer_bias
        # Apply sigmoid activation to get activations at the output layer
        output_layer_activations = self.sigmoid(output_layer_weighted_sums)

        return hidden_layer_activations, output_layer_activations

    def backward_pass(self, inputs, output_activations, hidden_layer_activations):
        # Initialise an empty list to store the deltas
        deltas = []

        # Calculate the delta for the output layer
        deltas.append((inputs[-1] - output_activations[0]) * (output_activations[0] * (1 - output_activations[0])))

        # Calculate the deltas for the hidden layer nodes
        for x in range(self.hidden_layer_size):
            # The delta value for each hidden layer node is calculated based on the deltas from the output layer
            deltas.append((hidden_layer_activations[x] * (1 - hidden_layer_activations[x])) * deltas[x] *
                          self.hidden_layer_biases[x])

        return deltas

    def update_weights(self, inputs, hidden_activations, deltas, learning_rate, momentum_rate):
        # Calculate weight changes without momentum
        output_weight_change = learning_rate * np.outer(hidden_activations, deltas[0])
        hidden_weight_change = learning_rate * np.outer(inputs, deltas[1:])

        # Apply momentum to weight changes
        output_weight_change += momentum_rate * self.previous_output_weight_change
        hidden_weight_change += momentum_rate * self.previous_hidden_weight_change

        # Update output layer weights and biases
        self.output_layer_weights += output_weight_change
        self.output_layer_bias += learning_rate * deltas[0]

        # Update hidden layer weights and biases
        self.hidden_layer_weights += hidden_weight_change

        # Update previous weight changes
        self.previous_output_weight_change = output_weight_change
        self.previous_hidden_weight_change = hidden_weight_change

    def calculate_mean_validation_error(self, validation_data):
        # Initialise an empty list to store the errors
        validation_errors = []

        # Iterate through the rows of the standardised validation data calculating the validation error
        for _, val_row in validation_data.iterrows():
            _, val_output_layer_activations = self.forward_pass(val_row.values)
            error = np.mean((val_row.iloc[-1] - val_output_layer_activations[0]) ** 2) ** 0.5
            validation_errors.append(error)

        return np.mean(validation_errors)

    # Activation function
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Update the learning rate but ensure it does not drop below the end learning rate
    @staticmethod
    def simulated_annealing(start_lr, end_lr, max_epochs, current_epoch):
        new_rate = ((end_lr + (start_lr - end_lr)) * (1 - (1 / (1 + np.exp(10 - ((20 * current_epoch) / max_epochs))))))
        if new_rate > end_lr:
            return new_rate
        else:
            return end_lr