def __fn(X, y):
    assert X.dtype == np.dtype(np.float64)
    assert X.shape == (150,) + (4,)
    assert np.all(np.isfinite(X))
    assert y.dtype == np.dtype(np.int64)
    assert y.shape == (150,)
    assert set(np.unique(y)) == set((0,) + (1,) + (2,))
    _0 = y.reshape((-1,))
    _1 = _0 == np.array(0)
    _2 = np.sum(_1)
    _3 = _0 == np.array(1)
    _4 = np.sum(_3)
    _5 = _0 == np.array(2)
    _6 = np.sum(_5)
    _7 = np.array((_2.item(),) + (_4.item(),) + (_6.item(),)).astype(np.dtype(np.float64))
    _8 = _7 / np.array(150.0)
    _9 = np.zeros((3,) + (4,), dtype=np.dtype(np.float64))
    _10 = _9
    _11 = np.sum(X[_1], axis=0)
    _12 = _11 / np.array(X[_1].shape[0])
    _10[0, slice(None, None, None)] = _12
    _13 = _10
    _14 = np.sum(X[_3], axis=0)
    _15 = _14 / np.array(X[_3].shape[0])
    _13[1, slice(None, None, None)] = _15
    _16 = _13
    _17 = np.sum(X[_5], axis=0)
    _18 = _17 / np.array(X[_5].shape[0])
    _16[2, slice(None, None, None)] = _18
    _19 = _8 @ _16
    _20 = X - _19
    _21 = np.sqrt(np.array((1.0 / 147)))
    _22 = _0 == np.array((0,) + (1,) + (2,))[0]
    _23 = X[_22] - _16[0, slice(None, None, None)]
    _24 = _0 == np.array((0,) + (1,) + (2,))[1]
    _25 = X[_24] - _16[1, slice(None, None, None)]
    _26 = _0 == np.array((0,) + (1,) + (2,))[2]
    _27 = X[_26] - _16[2, slice(None, None, None)]
    _28 = np.concatenate((_23,) + (_25,) + (_27,), axis=0)
    _29 = np.sum(_28, axis=0)
    _30 = _29 / np.array(_28.shape[0])
    _31 = np.expand_dims(_30, 0)
    _32 = _28 - _31
    _33 = np.square(_32)
    _34 = np.sum(_33, axis=0)
    _35 = _34 / np.array(_33.shape[0])
    _36 = np.sqrt(_35)
    _37 = _36
    _38 = _36 == np.array(0)
    _37[_38] = np.array(1.0)
    _39 = _28 / _37
    _40 = _21 * _39
    _41 = np.linalg.svd(_40, full_matrices=False)
    _42 = _41[1] > np.array(0.0001)
    _43 = np.sum(_42)
    _44 = _43.astype(np.dtype(np.int32))
    _45 = _41[2][slice(None, _44.item(), None), slice(None, None, None)] / _37
    _46 = _45.T / _41[1][slice(None, _44.item(), None)]
    _47 = np.sqrt(np.array(((150 * _8.item()) * (1.0 / 2))))
    _48 = _16 - _19
    _49 = _47 * _48.T
    _50 = _49.T @ _46
    _51 = np.linalg.svd(_50, full_matrices=False)
    _52 = np.array(0.0001) * _51[1][0]
    _53 = _51[1] > _52
    _54 = np.sum(_53)
    _55 = _54.astype(np.dtype(np.int32))
    _56 = _46 @ _51[2].T[slice(None, None, None), slice(None, _55.item(), None)]
    _57 = _20 @ _56
    return _57[slice(None, None, None), slice(None, 2, None)]
