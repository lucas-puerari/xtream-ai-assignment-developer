import numpy as np

from optimizer import optmize_xgboost_model


mock_x_train = np.array([[1], [2], [3]])
mock_y_train = np.array([2, 3, 4])
mock_x_test = np.array([[1], [2], [3], [4], [5]])
mock_y_test = np.array([2, 3, 4, 5, 6])


def test_optmize_xgboost_model():
    result = optmize_xgboost_model(mock_x_train, mock_x_test, mock_y_train, mock_y_test)

    assert isinstance(result, dict)
