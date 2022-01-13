from Kozinets_algorithm import classificator, for_test
import pandas as pd
import numpy as np
import pytest

@pytest.mark.parametrize("X, y, beta",  [(np.array([[10, 14, 28, 19], [1, 2, 3, 4]]), np.array([1, 1]), np.zeros(7)),
                                        (np.array([10, 14, 28]), np.array([1]), np.zeros(5))])
def test_params_ValueError(X, y, beta):
    with pytest.raises(ValueError):
        classificator(X, y, beta)

@pytest.mark.parametrize("X, y, beta",  [(np.array([[10, 14, 28], [1, 2, 3, 4]]), np.array([1, 1]), np.zeros(7))])
def test_params_IndexError(X, y, beta):
    with pytest.raises(IndexError):
        classificator(X, y, beta)

@pytest.mark.parametrize("X, y, beta",  [(np.zeros((3, 2)), np.array([1, -1, 1]), np.zeros(7))])
def test_params_y_klass(X, y, beta):
    for i in range(len(y)):
        assert y[i] == 1 or -1


@pytest.mark.parametrize("X, y, beta, z",  [(np.zeros(7), np.array(1), np.zeros(7), 34)])
def test_params_TypeError(X, y, beta, z):
    with pytest.raises(TypeError):
        classificator(X, y, beta, z)

@pytest.mark.parametrize("df, beta",  [(pd.DataFrame({'X': [1, 2, 3], 'Z': [1, 2, 3]}) , np.zeros(7))])
def test_for_test_params_AtributeError(df, beta):
    with pytest.raises(AttributeError):
        for_test(df, beta)

@pytest.mark.parametrize("df, beta",  [(pd.DataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]}) , np.zeros(5))])
def test_for_test_params_ValueError(df, beta):
    with pytest.raises(ValueError):
        for_test(df, beta)


@pytest.mark.parametrize("df, beta, z",  [(pd.DataFrame({'X': [1, 2, 3], 'Y': [1, 2, 3]}) , np.zeros(5), 2)])
def test_for_test_params_TypeError(df, beta, z):
    with pytest.raises(TypeError):
        for_test(df, beta, z)