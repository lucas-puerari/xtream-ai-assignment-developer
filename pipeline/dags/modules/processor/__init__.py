import io
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_TEST_SPLIT = 0.2


def pre_data_preparation(csv_string):
    df_diamonds = pd.read_csv(io.StringIO(csv_string))

    df_diamonds = df_diamonds[
        (df_diamonds.x * df_diamonds.y * df_diamonds.z != 0)
        & (df_diamonds.price > 0)
    ]
    df_diamonds = df_diamonds.drop(columns=['depth', 'table', 'y', 'z'])

    price = df_diamonds.pop('price')
    df_diamonds['price'] = price

    return [df_diamonds.columns.tolist()] + df_diamonds.values.tolist()


def linear_data_preparation(diamonds):
    df_diamonds = pd.DataFrame(diamonds[1:], columns=diamonds[0])

    df_diamonds = pd.get_dummies(df_diamonds, columns=['cut', 'color', 'clarity'], drop_first=True)

    return [df_diamonds.columns.tolist()] + df_diamonds.values.tolist()


def post_data_preparation(diamonds):
    df_diamonds = pd.DataFrame(diamonds[1:], columns=diamonds[0])

    x = np.array(df_diamonds.drop(columns='price'))
    y = np.log(np.array(df_diamonds.price, dtype=float))

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TRAIN_TEST_SPLIT,
        random_state=SEED
    )

    return x_train.tolist(), x_test.tolist(), y_train.tolist(), y_test.tolist()
