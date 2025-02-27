def __fn(X, y):
    assert X.dtype == np.dtype(np.float64)
    assert X.shape == (150, 4, )
    assert np.all(np.isfinite(X))
    assert y.dtype == np.dtype(np.int64)
    assert y.shape == (150, )
    assert set(np.unique(y)) == set((0, 1, 2, ))
    _0 = y == np.array(0)
    _1 = np.sum(_0)
    _2 = y == np.array(1)
    _3 = np.sum(_2)
    _4 = y == np.array(2)
    _5 = np.sum(_4)
    _6 = np.array((_1, _3, _5, )).astype(np.dtype(np.float64))
    _7 = _6 / np.array(float(150))
    _8 = np.zeros((3, 4, ), dtype=np.dtype(np.float64))
    _9 = np.sum(X[_0], axis=0)
    _10 = _9 / np.array(X[_0].shape[0])
    _8[0, :,] = _10
    _11 = np.sum(X[_2], axis=0)
    _12 = _11 / np.array(X[_2].shape[0])
    _8[1, :,] = _12
    _13 = np.sum(X[_4], axis=0)
    _14 = _13 / np.array(X[_4].shape[0])
    _8[2, :,] = _14
    _15 = _7 @ _8
    _16 = X - _15
    _17 = np.sqrt(np.asarray(np.array(float(1 / 147)), np.dtype(np.float64)))
    _18 = X[_0] - _8[0, :,]
    _19 = X[_2] - _8[1, :,]
    _20 = X[_4] - _8[2, :,]
    _21 = np.concatenate((_18, _19, _20, ), axis=0)
    _22 = np.sum(_21, axis=0)
    _23 = _22 / np.array(_21.shape[0])
    _24 = np.expand_dims(_23, 0)
    _25 = _21 - _24
    _26 = np.square(_25)
    _27 = np.sum(_26, axis=0)
    _28 = _27 / np.array(_26.shape[0])
    _29 = np.sqrt(_28)
    _30 = _29 == np.array(0)
    _29[_30] = np.array(float(1))
    _31 = _21 / _29
    _32 = _17 * _31
    _33 = np.linalg.svd(_32, full_matrices=False)
    _34 = _33[1] > np.array(0.0001)
    _35 = _34.astype(np.dtype(np.int32))
    _36 = np.sum(_35)
    _37 = _33[2][:_36, :,] / _29
    _38 = _37.T / _33[1][:_36]
    _39 = np.array(150) * _7
    _40 = _39 * np.array(float(1 / 2))
    _41 = np.sqrt(_40)
    _42 = _8 - _15
    _43 = _41 * _42.T
    _44 = _43.T @ _38
    _45 = np.linalg.svd(_44, full_matrices=False)
    _46 = np.array(0.0001) * _45[1][0]
    _47 = _45[1] > _46
    _48 = _47.astype(np.dtype(np.int32))
    _49 = np.sum(_48)
    _50 = _38 @ _45[2].T[:, :_49,]
    _51 = _16 @ _50
    return _51[:, :2,]
