import pandas as pd


def scale_series(series, min_value, max_value, scaling_factor=0.8, offset=0.1):
    if min_value == max_value:
        return pd.Series([offset] * len(series), index=series.index)
    return (
        scaling_factor * (series - min_value) / (max_value - min_value) + offset
    )


def scale_dataframe(df, reference_df=None, scaling_factor=0.8, offset=0.1):
    """
    Scale all columns in a DataFrame based on min/max values.
    If reference_df is provided, use its min/max values (usually training set).
    """
    reference_df = df if reference_df is None else reference_df
    scaled_df = pd.DataFrame(index=df.index)

    for column in df.columns:
        min_val = reference_df[column].min()
        max_val = reference_df[column].max()
        scaled_df[column] = scale_series(
            df[column], min_val, max_val, scaling_factor, offset
        )

    return scaled_df


def standardise_data(x_train, x_val, x_test, y_train, y_val, y_test):
    scaling_factor = 0.8

    x_train_standardised = scale_dataframe(x_train)
    x_val_standardised = scale_dataframe(x_val, reference_df=x_train)
    x_test_standardised = scale_dataframe(x_test, reference_df=x_train)

    y_min = y_train.min()
    y_max = y_train.max()
    y_train_standardised = scale_series(y_train, y_min, y_max, scaling_factor)
    y_val_standardised = scale_series(y_val, y_min, y_max, scaling_factor)
    y_test_standardised = scale_series(y_test, y_min, y_max, scaling_factor)

    input_size = len(x_train.columns)

    return (
        x_train_standardised,
        x_val_standardised,
        x_test_standardised,
        y_train_standardised,
        y_val_standardised,
        y_test_standardised,
        input_size,
    )
