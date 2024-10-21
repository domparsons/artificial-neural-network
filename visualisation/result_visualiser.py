import plotly.graph_objects as go
import plotly.express as px

class ResultVisualiser:

    @staticmethod
    def show_scatter_plot(y_test, y_predict):
        fig = px.scatter(x=y_test, y=y_predict, labels={'x': 'Actual', 'y': 'Predicted'}, title='Actual vs Predicted')
        fig.update_traces(marker=dict(size=8, color='blue', opacity=0.6))
        fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted')
        fig.show()

    @staticmethod
    def show_validation_error_plot(epochs, epoch_split, mean_validation_errors):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(0, epochs, epoch_split)),
                                 y=mean_validation_errors,
                                 mode='lines+markers',
                                 line=dict(color='firebrick', width=2),
                                 marker=dict(size=8),
                                 name='Mean Validation Error'))
        fig.update_layout(title='Mean Validation Errors Throughout Training',
                          xaxis_title='Epochs',
                          yaxis_title='Mean Validation Error')
        fig.show()

    @staticmethod
    def print_prediction_data(correlation_coefficient, precision, threshold, mae):
        print("\nPrediction data:")
        print(f"Correlation: {correlation_coefficient*100:.2f}%")
        print(f"Precision (within {threshold}): {precision*100:.2f}%")
        print("Mean Absolute Error:", mae)