import numpy as np
from scipy.odr import odr


def linear_rigid_body_dynamics(regressors, iparams):
    # Not sure but odr would expect regressors's shape to be (num_params, num_samples) rather than (num_samples, num_params)
    return regressors.T @ iparams


def total_lstsq(regressors: np.ndarray, fts_sen: np.ndarray, initbeta_scale: float = 1.0) -> tuple:
    """Perform Total Least Squares (TLS) regression. Since TLS regression is essentially the same as
    Orthogonal Distance Regression (ODR), scipy.odr's odr() method is used.

    Args:
        regressors (np.ndarray): The regressor matrix.
        fts_sen (np.ndarray): The sensor measurements.
        initbeta_scale (float, optional): The initial scale for the beta parameters. Defaults to 1.0.

    Returns:
        tuple: _description_
    """
    initbeta = initbeta_scale * np.ones(regressors.shape[-1])

    return odr(
        fcn=linear_rigid_body_dynamics,
        initbeta=initbeta,
        y=fts_sen,
        # Not sure but odr would expect x's shape to be (num_params, num_samples) rather than
        # (num_samples, num_params) like torch. So, configure the argument below in such a way
        x=regressors.T,
        full_output=1,
    )
