Help on method standard_normal in module numpy.random._generator:

ssttaannddaarrdd__nnoorrmmaall(size=None, dtype=<class 'numpy.float64'>, out=None) method of numpy.random._generator.Generator instance
    standard_normal(size=None, dtype=np.float64, out=None)

    Draw samples from a standard Normal distribution (mean=0, stdev=1).

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    dtype : dtype, optional
        Desired dtype of the result, only `float64` and `float32` are supported.
        Byteorder must be native. The default value is np.float64.
    out : ndarray, optional
        Alternative output array in which to place the result. If size is not None,
        it must have the same shape as the provided size and must match the type of
        the output values.

    Returns
    -------
    out : float or ndarray
        A floating-point array of shape ``size`` of drawn samples, or a
        single sample if ``size`` was not specified.

    See Also
    --------
    normal :
        Equivalent function with additional ``loc`` and ``scale`` arguments
        for setting the mean and standard deviation.

    Notes
    -----
    For random samples from the normal distribution with mean ``mu`` and
    standard deviation ``sigma``, use one of::

        mu + sigma * rng.standard_normal(size=...)
        rng.normal(mu, sigma, size=...)

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> rng.standard_normal()
    2.1923875335537315 # random

    >>> s = rng.standard_normal(8000)
    >>> s
    array([ 0.6888893 ,  0.78096262, -0.89086505, ...,  0.49876311,  # random
           -0.38672696, -0.4685006 ])                                # random
    >>> s.shape
    (8000,)
    >>> s = rng.standard_normal(size=(3, 4, 2))
    >>> s.shape
    (3, 4, 2)

    Two-by-four array of samples from the normal distribution with
    mean 3 and standard deviation 2.5:

    >>> 3 + 2.5 * rng.standard_normal(size=(2, 4))
    array([[-4.49401501,  4.00950034, -1.81814867,  7.29718677],   # random
           [ 0.39924804,  4.68456316,  4.99394529,  4.84057254]])  # random
