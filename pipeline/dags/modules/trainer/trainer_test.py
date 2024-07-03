import pickle
import numpy as np

from trainer import train_linear_model, evaluate_model


class MockModel:
    # pylint: disable=unused-argument
    def predict(self, x_test):
        return np.array([1, 2, 3, 4, 5])


def test_train_linear_model():
    mock_x_train = np.array([[1], [2], [3]])
    mock_y_train = np.array([2, 3, 4])

    result = train_linear_model(mock_x_train, mock_y_train)

    assert isinstance(result, bytes)


def test_evaluate_model():
    mock_model = pickle.dumps(MockModel())

    x_test = np.array([[1], [2], [3], [4], [5]])
    y_test = np.array([2, 3, 4, 5, 6])

    r2, mae = evaluate_model(mock_model, x_test, y_test)

    assert isinstance(r2, float)
    assert isinstance(mae, float)
