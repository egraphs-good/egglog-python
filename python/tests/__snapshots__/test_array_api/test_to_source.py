def __fn(X, y):
    _0 = np.sqrt(np.array((np.array(1.0).item() / 147)))
    assert X.dtype == np.float64
    assert X.shape == (150,) + (4,)
    assert np.all(np.isfinite(X))
    assert y.dtype == np.int64
    assert y.shape == (150,)
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    _1 = y.reshape((-1,))
    _2 = np.array((0,) + (1,) + (2,))
    _3 = np.zeros((3,) + (4,), dtype=np.float64)
    _4 = _3
    _5 = np.unique(_1, return_inverse=True)
    _6 = np.mean(X[_5[1] == np.array(0)], axis=0)
    _4[0, slice(None, None, None)] = _6
    _7 = _4
    _8 = np.mean(X[_5[1] == np.array(1)], axis=0)
    _7[1, slice(None, None, None)] = _8
    _9 = _7
    _10 = np.mean(X[_5[1] == np.array(2)], axis=0)
    _9[2, slice(None, None, None)] = _10
    _11 = X[_1 == _2[0]] - _9[0, slice(None, None, None)]
    _12 = X[_1 == _2[1]] - _9[1, slice(None, None, None)]
    _13 = X[_1 == _2[2]] - _9[2, slice(None, None, None)]
    _14 = np.concatenate((_11,) + (_12,) + (_13,), axis=0)
    _15 = np.std(_14, axis=0)
    _16 = _15
    _16[_15 == np.array(0)] = np.array(1.0)
    _17 = _14 / _16
    _18 = _0 * _17
    _19 = np.linalg.svd(_18, full_matrices=False)
    _20 = _19[1] > np.array(0.0001)
    _21 = np.sum(_20, axis=None)
    _22 = _21.astype(np.int32)
    _23 = _19[2][slice(None, _22.item(), None), slice(None, None, None)] / _16
    _24 = _23.T / _19[1][slice(None, _22.item(), None)]
    _25 = np.unique(_1, return_counts=True)
    _26 = _25[1].astype(np.float64)
    _27 = _26 / np.array(150.0)
    _28 = _24 + _27
    return _28
