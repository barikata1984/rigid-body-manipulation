import numpy as np
import pytest

from regressions import total_lstsq


def test_tls_recovers_noise_free_parameters():
    a = np.random.default_rng(42).normal(size=(300, 10))
    expected = np.array([1.12, 0, 0, 0.17, 0.03, 0.03, 0.001, 0, 0, 0])
    np.testing.assert_allclose(total_lstsq(a, a @ expected), expected, rtol=0, atol=1e-14)


def test_tls_corrects_errors_in_both_inputs():
    # The principal axis of [[1, 1], [0, 1]] has slope (1 + sqrt(5)) / 2.
    actual = total_lstsq(np.array([[1.0], [0.0]]), np.array([1.0, 1.0]))
    np.testing.assert_allclose(actual, [(1 + np.sqrt(5)) / 2], rtol=1e-14)


@pytest.mark.parametrize(
    "a,b",
    [
        (np.ones((3, 2)), np.ones(3)),  # unidentifiable parameters
        (np.array([[1.0], [0.0]]), np.array([0.0, 2.0])),  # no finite TLS solution
        (np.eye(2), np.ones(2)),  # not overdetermined
        (np.ones((3, 1)), np.ones(2)),  # mismatched observations
        (np.array([[np.nan], [1.0]]), np.ones(2)),
    ],
)
def test_tls_rejects_invalid_or_nonunique_problems(a, b):
    with pytest.raises(ValueError):
        total_lstsq(a, b)
