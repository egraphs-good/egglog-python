def my_fn(X, y):
    assert y.dtype == np.int64
    assert X.dtype == np.float64
    assert y.dtype == np.int64
    assert X.dtype == np.float64
    assert y.shape == (150,)
    assert y.shape == (150,)
    _0 = np.array(150.0)
    assert X.shape == (150,) + (4,)
    assert X.shape == (150,) + (4,)
    assert y.shape == (150,)
    assert y.shape == (150,)
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    _1 = y.reshape((-1,))
    _1 = y.reshape((-1,))
    _2 = np.unique(_1, return_counts=True)
    _2 = np.unique(_1, return_counts=True)
    _3 = np.unique(_1)
    _4 = _2[1].astype(np.float64)
    _4 = _2[1].astype(np.float64)
    _5 = _4 / _0
    _6 = _5 + X
    _7 = np.zeros((3,) + (4,), dtype=np.float64)
    _6 = _5 + X
    _7 = _6 + _7
    _7 = np.zeros((3,) + (4,), dtype=np.float64)
    _7 = _6 + _7
    return _7
