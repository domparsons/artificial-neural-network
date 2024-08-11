
# ********************************************* Imports ******************************************************************* #

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error

# ********************************************* End of imports ************************************************************ #



# ********************************************* Start of Data Processor class ********************************************* #

class DataProcessor:

    def load_and_process_data(self, file_path, sheet_name, columns):
        # Import dataset from Excel into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns).copy()
        df = self.clean_data(df)
        df = self.select_features(df, correlation_threshold=0.2, input_corr_threshold=0.95)
        
        return df

    def clean_data(self, df):
        # Convert non-numeric values to NaN and drop rows with NaN
        df = df.apply(pd.to_numeric, errors='coerce').dropna()

        # Drop rows with negative values
        df = df[(df >= 0).all(axis=1)]
        
        return df

    def select_features(self, df, correlation_threshold, input_corr_threshold):
        # Calculate correlation coefficients and remove columns with less than threshold value (default ±0.2)
        output_column = df.columns[-1]
        correlations = df.corr().iloc[:, -1]
        for col, corr in correlations.items():
            if col != output_column and abs(corr) < correlation_threshold:
                df = df.drop(columns=[col])

        # Calculate correlation matrix of coefficients between input columns
        correlations = df.corr().abs()
        upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool)) 

        # Remove columns with high correlation with other inputs (default threshold ±0.9)
        columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] >= input_corr_threshold)]
        columns_to_drop = [col for col in columns_to_drop if col != output_column]
        df = df.drop(columns=columns_to_drop)

        return df

# ********************************************* End of Data Processor class *********************************************** #
    


# ********************************************* Start of Data Splitter class ********************************************** #

class DataSplitter:

    def __init__(self, df):
        self.df = df

    def split_data(self, train_size=0.6, val_size=0.2):
        # Set a random seed for reproducibility
        np.random.seed(42)

        # Shuffle the indices of DataFrame to ensure random data is chosen for each set
        indices = np.random.permutation(self.df.index)

        # Define train, validate, test data split (60/20/20)
        train_size = int(train_size * len(self.df))
        val_size = int(val_size * len(self.df))
        test_size = len(self.df) - train_size - val_size
        
        # Split the indices into train, validate, and test sets
        train_indices = indices[:train_size]
        val_indices = indices[train_size:(train_size + val_size)]
        test_indices = indices[(train_size + val_size):]

        # Create training, validation, and testing data sets
        training_data = self.df.loc[train_indices]
        validation_data = self.df.loc[val_indices]
        testing_data = self.df.loc[test_indices]

        return training_data, validation_data, testing_data
    
# ********************************************* End of Data Splitter class ************************************************ #
    


# ********************************************* Start of Data Standardiser class ****************************************** #

class DataStandardiser:

    def __init__(self):
        pass

    def standardise_data(self, train_and_validate):
        # Create new empty data frames to hold standardised data
        standardised_training_data = pd.DataFrame()
        standardised_validation_data = pd.DataFrame()
        standardised_testing_data = pd.DataFrame()

        # Iterate through the columns and calculate the min and max
        for column in train_and_validate.columns:
            min_value = train_and_validate[column].min()
            max_value = train_and_validate[column].max()

            # Standardise the data using the calculated min and max for each set of data
            standardised_training_data[column] = training_data[column].apply(lambda x: self.get_standardised_value(x, min_value, max_value))
            standardised_validation_data[column] = validation_data[column].apply(lambda x: self.get_standardised_value(x, min_value, max_value))
            standardised_testing_data[column] = testing_data[column].apply(lambda x: self.get_standardised_value(x, min_value, max_value))

        return standardised_training_data, standardised_validation_data, standardised_testing_data

    # Standardising function
    def get_standardised_value(self, value, min_val, max_val, scaling_factor=0.8):
        scaled_value = scaling_factor * ((value - min_val) / (max_val - min_val)) + 0.1
        return scaled_value
    
# ********************************************* End of Data Standardiser class ******************************************** #



# ********************************************* Start of Neural Network class ********************************************* #

class NeuralNetwork:

    def __init__(self, input_size, hidden_layer_size, output_size):
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size
        
        # Initialise weights and biases for the hidden layer
        self.hidden_layer_weights = np.random.randn(input_size, hidden_layer_size) * 0.01 
        self.hidden_layer_biases = np.array([random.uniform(-0.1, 0.1) for _ in range(hidden_layer_size)])
        
        # Initialise weights and biases for the output layer
        self.output_layer_weights = np.random.randn(hidden_layer_size, output_size)
        self.output_layer_bias = random.uniform(-0.1, 0.1)
        
        # Store previous change in weights for momentum
        self.previous_output_weight_change = np.zeros_like(self.output_layer_weights)
        self.previous_hidden_weight_change = np.zeros_like(self.hidden_layer_weights)
        
    def train(self, training_data, validation_data, epochs, initial_learning_rate, final_learning_rate, momentum_rate, epoch_split):
        # Store mean validation errors in a variable to plot error change through epochs
        mean_validation_errors = []

        # Training loop
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
        output_layer_weighted_sums = np.dot(hidden_layer_activations, self.output_layer_weights) + self.output_layer_bias
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
            deltas.append((hidden_layer_activations[x] * (1 - hidden_layer_activations[x])) * deltas[x] * self.hidden_layer_biases[x])

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
            error = np.mean((val_row.iloc[-1] - val_output_layer_activations[0])**2)**0.5
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

# ********************************************* End of Neural Network class *********************************************** #



# ********************************************* Start of Flood Predictor class ******************************************** #

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
    
# ********************************************* End of Flood Predictor class ********************************************** #
    


# ********************************************* Start of Result Visualiser class ****************************************** #

class ResultVisualiser:

    # Create and show a scatter graph of the predicted and actual data
    @staticmethod
    def show_scatter_plot(y_test, y_predict):
        plt.scatter(y_test, y_predict)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.show()

    # Plot and show a line graph of the validation error
    @staticmethod
    def show_validation_error_plot(epochs, epoch_split, mean_validation_errors):
        plt.plot(range(0, epochs, epoch_split), mean_validation_errors, marker='o', linestyle='-')
        plt.xlabel('Epochs')
        plt.ylabel('Mean Validation Error')
        plt.title('Mean Validation Errors Throughout Training')
        plt.show()

    # Print the correlation and precision calculations
    @staticmethod
    def print_prediction_data(correlation_coefficient, precision, threshold, mae):
        print("\nPrediction data:")
        print(f"Correlation: {correlation_coefficient*100:.2f}%")
        print(f"Precision (within {threshold}): {precision*100:.2f}%")
        print("Mean Absolute Error:", mae)

# ********************************************* End of Result Visualiser class ******************************************** #
        


# ********************************************* Start of main program ***************************************************** #

# Load and preprocess data
data_processor = DataProcessor()
file_path = '/Users/domparsons/Documents/Loughborough/Year 2/AI Methods/ANN Coursework/FEHDataStudent.xlsx'
sheet_name = 'Sheet1'
columns = ['AREA', 'BFIHOST', 'FARL', 'FPEXT', 'LDP', 'PROPWET', 'RMED-1D', 'SAAR', 'Index flood']
df = data_processor.load_and_process_data(file_path, sheet_name, columns)

# Split the data
splitter = DataSplitter(df)
training_data, validation_data, testing_data = splitter.split_data()

# Standardise the data
train_and_validate = pd.concat([training_data, validation_data], axis=0)
standardiser = DataStandardiser()
standardised_training_data, standardised_validation_data, standardised_testing_data = standardiser.standardise_data(train_and_validate)
    
# Define hyperparameters
epochs = 400
epoch_split = 20
initial_learning_rate = 0.1
final_learning_rate = 0.01

# Define network structure
input_size = len(standardised_training_data.columns)
hidden_layer_size = 10
output_size = 1

# Define momentum rate 
momentum_rate = 0.8

# Train the model
neural_network = NeuralNetwork(input_size, hidden_layer_size, output_size)
mean_validation_errors, hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias = neural_network.train(standardised_training_data, standardised_validation_data, epochs, initial_learning_rate, final_learning_rate, momentum_rate, epoch_split)

# Predict index flood using test data
flood_predictor = FloodPredictor(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias)
y_predict = flood_predictor.predict(standardised_testing_data, train_and_validate)
y_test = testing_data['Index flood'].values
correlation_coefficient, precision, mae = FloodPredictor.evaluate(y_test, y_predict)

# Visualise the results
threshold = 60
ResultVisualiser.show_scatter_plot(y_test, y_predict)
ResultVisualiser.show_validation_error_plot(epochs, epoch_split, mean_validation_errors)
ResultVisualiser.print_prediction_data(correlation_coefficient, precision, threshold, mae)

# Read LINEST actual and prediction data into a DataFrame
file_path = '/Users/domparsons/Documents/Loughborough/Year 2/AI Methods/ANN Coursework/compare-data.xlsx'
columns = ['Index flood', 'Pred']
df = pd.read_excel(file_path, usecols=columns)
y_test = df['Index flood'].values
y_predict = df['Pred'].values

# Display the scatter with LINEST data
ResultVisualiser.show_scatter_plot(y_test, y_predict)

# Calculate and print evaluation
correlation_coefficient, precision, mae = FloodPredictor.evaluate(y_test, y_predict)
ResultVisualiser.print_prediction_data(correlation_coefficient, precision, threshold, mae)

# ********************************************* End of main program ******************************************************* #
