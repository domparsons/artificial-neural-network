import pandas as pd
import numpy as np

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
        columns_to_drop = [column for column in upper_triangle.columns if
                           any(upper_triangle[column] >= input_corr_threshold)]
        columns_to_drop = [col for col in columns_to_drop if col != output_column]
        df = df.drop(columns=columns_to_drop)

        return df