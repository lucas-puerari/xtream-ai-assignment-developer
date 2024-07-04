import pickle
import numpy as np

from trainer import train_linear_model, train_xgboost_model, evaluate_model


mock_x_train = np.array([[1], [2], [3]])
mock_y_train = np.array([2, 3, 4])
mock_x_test = np.array([[1], [2], [3], [4], [5]])
mock_y_test = np.array([2, 3, 4, 5, 6])
mock_prediction = np.array([1, 2, 3, 4, 5])
params = {}


class MockModel:
    # pylint: disable=unused-argument
    def predict(self, x_test):
        return mock_prediction

    def get_params(self):
        return {}


def test_train_linear_model():
    result = train_linear_model(mock_x_train, mock_y_train)

    assert isinstance(result, bytes)


def test_train_xgboost_model():
    result = train_xgboost_model(mock_x_train, mock_y_train, params)

    assert isinstance(result, bytes)


def test_evaluate_model():
    mock_model = pickle.dumps(MockModel())

    r2, mae = evaluate_model(mock_model, mock_x_test, mock_y_test)

    assert isinstance(r2, float)
    assert isinstance(mae, float)
