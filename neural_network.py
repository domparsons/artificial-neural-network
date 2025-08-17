from config.config import FilePaths, Hyperparameters
from models.flood_predictor import FloodPredictor
from models.neural_network import NeuralNetwork
from processing.data_processor import load_data, process_data
from processing.data_splitter import split_data
from processing.data_standardiser import standardise_data
from visualisation.result_visualiser import ResultVisualiser


def main():
    df = load_data(FilePaths.data_file, FilePaths.sheet_name, FilePaths.columns)
    target_col = df.columns.tolist()[-1]
    df = process_data(df, target_col)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(df, target_col)

    (
        x_train_standardised,
        x_val_standardised,
        x_test_standardised,
        y_train_standardised,
        y_val_standardised,
        y_test_standardised,
        input_size,
    ) = standardise_data(x_train, x_val, x_test, y_train, y_val, y_test)

    neural_network = NeuralNetwork(
        input_size,
        Hyperparameters.hidden_layer_size,
        Hyperparameters.output_size,
    )

    (
        mean_validation_errors,
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_bias,
    ) = neural_network.train(
        x_train_standardised,
        x_val_standardised,
        y_train_standardised,
        y_val_standardised,
        Hyperparameters.epochs,
        Hyperparameters.initial_learning_rate,
        Hyperparameters.final_learning_rate,
        Hyperparameters.momentum_rate,
    )

    flood_predictor = FloodPredictor(
        hidden_layer_weights,
        hidden_layer_biases,
        output_layer_weights,
        output_layer_bias,
    )
    y_predict = flood_predictor.predict(x_test_standardised, y_test)
    y_test = y_test.values

    correlation_coefficient, precision, mae = FloodPredictor.evaluate(
        y_test, y_predict
    )
    threshold = 60

    ResultVisualiser.show_scatter_plot(y_test, y_predict)
    ResultVisualiser.show_validation_error_plot(
        Hyperparameters.epochs,
        mean_validation_errors,
    )
    ResultVisualiser.print_prediction_data(
        correlation_coefficient, precision, threshold, mae
    )


if __name__ == "__main__":
    main()
