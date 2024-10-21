import pandas as pd

from config.config import FilePaths, Hyperparameters
from models.flood_predictor import FloodPredictor
from models.neural_network import NeuralNetwork
from processing.data_processor import DataProcessor
from processing.data_splitter import DataSplitter
from processing.data_standardiser import DataStandardiser
from visualisation.result_visualiser import ResultVisualiser


def main():
    data_processor = DataProcessor()
    df = data_processor.load_and_process_data(FilePaths.data_file, FilePaths.sheet_name, FilePaths.columns)

    splitter = DataSplitter(df)
    training_data, validation_data, testing_data = splitter.split_data()

    train_and_validate = pd.concat([training_data, validation_data], axis=0)
    standardiser = DataStandardiser()
    standardised_training_data, standardised_validation_data, standardised_testing_data = standardiser.standardise_data(
        train_and_validate, training_data, validation_data, testing_data
    )

    input_size = len(standardised_training_data.columns)
    neural_network = NeuralNetwork(input_size, Hyperparameters.hidden_layer_size, Hyperparameters.output_size)
    mean_validation_errors, hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias = neural_network.train(
        standardised_training_data, standardised_validation_data, Hyperparameters.epochs, 
        Hyperparameters.initial_learning_rate, Hyperparameters.final_learning_rate, 
        Hyperparameters.momentum_rate, Hyperparameters.epoch_split
    )

    flood_predictor = FloodPredictor(hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_bias)
    y_predict = flood_predictor.predict(standardised_testing_data, train_and_validate)
    y_test = testing_data['Index flood'].values

    correlation_coefficient, precision, mae = FloodPredictor.evaluate(y_test, y_predict)
    threshold = 60

    ResultVisualiser.show_scatter_plot(y_test, y_predict)
    ResultVisualiser.show_validation_error_plot(Hyperparameters.epochs, Hyperparameters.epoch_split, mean_validation_errors)
    ResultVisualiser.print_prediction_data(correlation_coefficient, precision, threshold, mae)

    compare_with_linest(FilePaths.compare_file, threshold)


def compare_with_linest(file_path, threshold):
    columns = ['Index flood', 'Pred']
    df = pd.read_excel(file_path, usecols=columns)
    y_test = df['Index flood'].values
    y_predict = df['Pred'].values

    ResultVisualiser.show_scatter_plot(y_test, y_predict)

    correlation_coefficient, precision, mae = FloodPredictor.evaluate(y_test, y_predict)
    ResultVisualiser.print_prediction_data(correlation_coefficient, precision, threshold, mae)


if __name__ == "__main__":
    main()