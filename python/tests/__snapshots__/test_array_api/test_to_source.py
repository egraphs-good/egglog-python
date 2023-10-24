def __fn(X, y):
    assert X.dtype == np.dtype(np.float64)
    assert X.shape == (150,) + (4,)
    assert np.all(np.isfinite(X))
    assert y.dtype == np.dtype(np.int64)
    assert y.shape == (150,)
    assert set(np.unique(y)) == set((0,) + (1,) + (2,))
    _0 = y == np.array(0)
    _1 = np.sum(_0)
    _2 = y == np.array(1)
    _3 = np.sum(_2)
    _4 = y == np.array(2)
    _5 = np.sum(_4)
    _6 = np.array((_1.item(),) + (_3.item(),) + (_5.item(),)).astype(np.dtype(np.float64))
    _7 = _6 / np.array(150.0)
    _8 = np.zeros((3,) + (4,), dtype=np.dtype(np.float64))
    _9 = np.sum(X[_0], axis=0)
    _10 = _9 / np.array(X[_0].shape[0])
    _8[0, :] = _10
    _11 = np.sum(X[_2], axis=0)
    _12 = _11 / np.array(X[_2].shape[0])
    _8[1, :] = _12
    _13 = np.sum(X[_4], axis=0)
    _14 = _13 / np.array(X[_4].shape[0])
    _8[2, :] = _14
    _15 = _7 @ _8
    _16 = X - _15
    _17 = np.sqrt(np.array(0.006802721088435374))
    _18 = y == np.array((0,) + (1,) + (2,))[0]
    _19 = X[_18] - _8[0, :]
    _20 = y == np.array((0,) + (1,) + (2,))[1]
    _21 = X[_20] - _8[1, :]
    _22 = y == np.array((0,) + (1,) + (2,))[2]
    _23 = X[_22] - _8[2, :]
    _24 = np.concatenate((_19,) + (_21,) + (_23,), axis=0)
    _25 = np.sum(_24, axis=0)
    _26 = _25 / np.array(_24.shape[0])
    _27 = np.expand_dims(_26, 0)
    _28 = _24 - _27
    _29 = np.square(_28)
    _30 = np.sum(_29, axis=0)
    _31 = _30 / np.array(_29.shape[0])
    _32 = np.sqrt(_31)
    _33 = _32 == np.array(0)
    _32[_33] = np.array(1.0)
    _34 = _24 / _32
    _35 = _17 * _34
    _36 = np.linalg.svd(_35, full_matrices=False)
    _37 = _36[1] > np.array(0.0001)
    _38 = _37.astype(np.dtype(np.int32))
    _39 = np.sum(_38)
    _40 = _36[2][:_39.item(), :] / _32
    _41 = _40.T / _36[1][:_39.item()]
    _42 = np.array(150) * _7
    _43 = _42 * np.array(0.5)
    _44 = np.sqrt(_43)
    _45 = _8 - _15
    _46 = _44 * _45.T
    _47 = _46.T @ _41
    _48 = np.linalg.svd(_47, full_matrices=False)
    _49 = np.array(0.0001) * _48[1][0]
    _50 = _48[1] > _49
    _51 = _50.astype(np.dtype(np.int32))
    _52 = np.sum(_51)
    _53 = _41 @ _48[2].T[:, :_52.item()]
    _54 = _16 @ _53
    return _54[:, :2]
