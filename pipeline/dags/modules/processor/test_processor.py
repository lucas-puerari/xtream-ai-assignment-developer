from processor import pre_data_preparation, linear_data_preparation, post_data_preparation


def test_pre_data_preparation():
    mock_data = """
carat,cut,color,clarity,depth,table,price,x,y,z
1.1,Ideal,H,SI2,62.0,55.0,4733,6.61,6.65,4.11
0.3,Very Good,H,IF,62.9,58.0,789,4.26,4.29,2.69
1.98,Premium,VVS2,SI2,61.2,57.2,0,9.02,3.23,5.34
"""

    result = pre_data_preparation(mock_data)

    assert isinstance(result, list)
    assert len(result) == 3
    assert len(result[0]) == 6


def test_linear_data_preparation():
    mock_data = [
        ['carat', 'cut', 'color', 'clarity', 'depth', 'x', 'price'],
        [1.1, 'Ideal', 'H', 'SI2', 62.0, 6.61, 4733],
        [0.9, 'Premium', 'E', 'VS1', 61.5, 6.14, 4500],
        [1.5, 'Good', 'G', 'VVS2', 63.0, 7.05, 9000],
        [0.7, 'Fair', 'J', 'SI1', 64.0, 5.5, 3000],
        [1.2, 'Very Good', 'I', 'VS2', 62.5, 6.75, 5200]
    ]

    result = linear_data_preparation(mock_data)

    assert isinstance(result, list)
    assert len(result) == 6
    assert len(result[0]) == 16


def test_post_data_preparation():
    mock_data = [
        ['carat', 'cut', 'color', 'clarity', 'depth', 'x', 'price'],
        [1.1, 'Ideal', 'H', 'SI2', 62.0, 6.61, 4733],
        [0.9, 'Premium', 'E', 'VS1', 61.5, 6.14, 4500],
        [1.5, 'Good', 'G', 'VVS2', 63.0, 7.05, 9000],
        [0.7, 'Fair', 'J', 'SI1', 64.0, 5.5, 3000],
        [1.2, 'Very Good', 'I', 'VS2', 62.5, 6.75, 5200]
    ]

    x_train, x_test, y_train, y_test = post_data_preparation(mock_data)

    assert isinstance(x_train, list)
    assert isinstance(y_train, list)
    assert isinstance(x_test, list)
    assert isinstance(y_test, list)

    assert len(x_train) == 4
    assert len(y_train) == 4
    assert len(x_test) == 1
    assert len(y_test) == 1

    assert len(x_train[0]) == 6
    assert isinstance(y_train[0], float)
