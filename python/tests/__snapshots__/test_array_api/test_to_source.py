def __fn(X, y):
    assert y.dtype == np.int64
    assert y.shape == (150,)
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    _0 = y.reshape((-1,))
    _1 = np.zeros((3,) + (4,), dtype=np.float64)
    _2 = _0 + _1
    _3 = np.unique(_0, return_counts=True)
    _4 = _3[1].astype(np.float64)
    _5 = _4 / np.array(150.0)
    _6 = _2 + _5
    return _6
