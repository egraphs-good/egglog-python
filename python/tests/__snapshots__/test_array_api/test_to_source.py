def __fn(X, y):
    assert X.dtype == np.float64
    assert X.shape == (150,) + (4,)
    assert np.all(np.isfinite(X))
    assert y.dtype == np.int64
    assert y.shape == (150,)
    assert set(y.flatten()) == set((0,) + (1,) + (2,))
    _0 = y.reshape((-1,))
    _1 = np.unique(_0, return_counts=True)
    _2 = _1[1].astype(np.float64)
    _3 = _2 / np.array(150.0)
    _4 = np.zeros((3,) + (4,), dtype=np.float64)
    _5 = _4
    _6 = np.unique(_0, return_inverse=True)
    _7 = _6[1] == np.array(0)
    _8 = np.mean(X[_7], axis=0)
    _5[0, slice(None, None, None)] = _8
    _9 = _5
    _10 = _6[1] == np.array(1)
    _11 = np.mean(X[_10], axis=0)
    _9[1, slice(None, None, None)] = _11
    _12 = _9
    _13 = _6[1] == np.array(2)
    _14 = np.mean(X[_13], axis=0)
    _12[2, slice(None, None, None)] = _14
    _15 = _3 @ _12
    _16 = X - _15
    _17 = np.sqrt(np.array((np.array(1.0).item() / 147)))
    _18 = _0 == np.array((0,) + (1,) + (2,))[0]
    _19 = X[_18] - _12[0, slice(None, None, None)]
    _20 = _0 == np.array((0,) + (1,) + (2,))[1]
    _21 = X[_20] - _12[1, slice(None, None, None)]
    _22 = _0 == np.array((0,) + (1,) + (2,))[2]
    _23 = X[_22] - _12[2, slice(None, None, None)]
    _24 = np.concatenate((_19,) + (_21,) + (_23,), axis=0)
    _25 = np.std(_24, axis=0)
    _26 = _25
    _27 = _25 == np.array(0)
    _26[_27] = np.array(1.0)
    _28 = _24 / _26
    _29 = _17 * _28
    _30 = np.linalg.svd(_29, full_matrices=False)
    _31 = _30[1] > np.array(0.0001)
    _32 = np.sum(_31, axis=None)
    _33 = _32.astype(np.int32)
    _34 = _30[2][slice(None, _33.item(), None), slice(None, None, None)] / _26
    _35 = _34.T / _30[1][slice(None, _33.item(), None)]
    _36 = np.sqrt(np.array(((150 * _3.item()) * (np.array(1.0).item() / 2))))
    _37 = _12 - _15
    _38 = _36 * _37.T
    _39 = _38.T @ _35
    _40 = np.linalg.svd(_39, full_matrices=False)
    _41 = np.array(0.0001) * _40[1][0]
    _42 = _40[1] > _41
    _43 = np.sum(_42, axis=None)
    _44 = _43.astype(np.int32)
    _45 = _35 @ _40[2].T[slice(None, None, None), slice(None, _44.item(), None)]
    _46 = _16 @ _45
    return _46[slice(None, None, None), slice(None, 2, None)]
