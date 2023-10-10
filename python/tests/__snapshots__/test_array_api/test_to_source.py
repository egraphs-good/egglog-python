def __fn(X, y):
    _0 = np.zeros((3,) + (4,), dtype=np.float64)
    _1 = _0
    assert X.dtype == np.float64
    assert X.shape == (150,) + (4,)
    assert np.all(np.isfinite(X))
    assert y.dtype == np.int64
    assert y.shape == (150,)
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    _2 = y.reshape((-1,))
    _3 = np.unique(_2, return_inverse=True)
    _4 = np.mean(X[_3[1] == np.array(0)], axis=0)
    _1[0, slice(None, None, None)] = _4
    _5 = _1
    _6 = np.mean(X[_3[1] == np.array(1)], axis=0)
    _5[1, slice(None, None, None)] = _6
    _7 = _5
    _8 = np.mean(X[_3[1] == np.array(2)], axis=0)
    _7[2, slice(None, None, None)] = _8
    _9 = np.unique(_2, return_counts=True)
    _10 = _9[1].astype(np.float64)
    _11 = _10 / np.array(150.0)
    _12 = _7 + _11
    return _12
