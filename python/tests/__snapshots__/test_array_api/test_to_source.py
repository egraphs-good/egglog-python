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
    _33 = np.abs(_32)
    _34 = np.square(_33)
    _35 = np.sum(_34, axis=0)
    _36 = _35 / np.array(_34.shape[0])
    _37 = np.sqrt(_36)
    _38 = _37
    _39 = _37 == np.array(0)
    _38[_39] = np.array(1.0)
    _40 = _28 / _38
    _41 = _21 * _40
    _42 = np.linalg.svd(_41, full_matrices=False)
    _43 = _42[1] > np.array(0.0001)
    _44 = np.sum(_43)
    _45 = _44.astype(np.dtype(np.int32))
    _46 = _42[2][slice(None, _45.item(), None), slice(None, None, None)] / _38
    _47 = _46.T / _42[1][slice(None, _45.item(), None)]
    _48 = np.sqrt(np.array(((150 * _8.item()) * (1.0 / 2))))
    _49 = _16 - _19
    _50 = _48 * _49.T
    _51 = _50.T @ _47
    _52 = np.linalg.svd(_51, full_matrices=False)
    _53 = np.array(0.0001) * _52[1][0]
    _54 = _52[1] > _53
    _55 = np.sum(_54)
    _56 = _55.astype(np.dtype(np.int32))
    _57 = _47 @ _52[2].T[slice(None, None, None), slice(None, _56.item(), None)]
    _58 = _20 @ _57
    return _58[slice(None, None, None), slice(None, 2, None)]
