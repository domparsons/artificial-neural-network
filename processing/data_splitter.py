import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    x = df.drop(columns=target_col)
    y = df[target_col]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=(test_size + val_size), random_state=random_state
    )
    val_ratio = val_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=val_ratio, random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test
