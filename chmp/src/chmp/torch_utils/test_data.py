import numpy as np
import pytest

from .data import get_number_of_samples


def test_assert_consistent_shape():
    with pytest.raises(ValueError):
        get_number_of_samples([])

    with pytest.raises(ValueError):
        get_number_of_samples([None, None])

    with pytest.raises(ValueError):
        get_number_of_samples([np.zeros([10]), np.zeros([20])])

    assert get_number_of_samples([np.zeros([10]), np.zeros([10, 20])]) == 10
    assert get_number_of_samples([None, np.zeros([10]), np.zeros([10, 20])]) == 10
