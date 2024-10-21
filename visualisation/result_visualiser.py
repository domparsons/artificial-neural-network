import matplotlib.pyplot as plt

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