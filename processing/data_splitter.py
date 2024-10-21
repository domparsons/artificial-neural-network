import numpy as np

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