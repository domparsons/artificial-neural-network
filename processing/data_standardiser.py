import pandas as pd

class DataStandardiser:

    def __init__(self):
        pass

    def standardise_data(self, train_and_validate, training_data, validation_data, testing_data):
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