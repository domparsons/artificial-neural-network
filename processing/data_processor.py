import pandas as pd
import numpy as np

class DataProcessor:

    def load_and_process_data(self, file_path, sheet_name, columns):
        df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=columns).copy()
        df = self.clean_data(df)
        df = self.select_features(df, correlation_threshold=0.2, input_corr_threshold=0.95)

        return df

    @staticmethod
    def clean_data(df):
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        df = df[(df >= 0).all(axis=1)]

        return df

    @staticmethod
    def select_features(df, correlation_threshold, input_corr_threshold):
        output_column = df.columns[-1]
        correlations = df.corr().iloc[:, -1]
        for col, corr in correlations.items():
            if col != output_column and abs(corr) < correlation_threshold:
                df = df.drop(columns=[col])

        correlations = df.corr().abs()
        upper_triangle = correlations.where(np.triu(np.ones(correlations.shape), k=1).astype(bool))

        columns_to_drop = [column for column in upper_triangle.columns if
                           any(upper_triangle[column] >= input_corr_threshold)]
        columns_to_drop = [col for col in columns_to_drop if col != output_column]
        df = df.drop(columns=columns_to_drop)

        return df