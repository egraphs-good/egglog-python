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
    _9 = _8
    _10 = np.sum(X[_0], axis=0)
    _11 = _10 / np.array(X[_0].shape[0])
    _9[0, :] = _11
    _12 = _9
    _13 = np.sum(X[_2], axis=0)
    _14 = _13 / np.array(X[_2].shape[0])
    _12[1, :] = _14
    _15 = _12
    _16 = np.sum(X[_4], axis=0)
    _17 = _16 / np.array(X[_4].shape[0])
    _15[2, :] = _17
    _18 = _7 @ _15
    _19 = X - _18
    _20 = np.sqrt(np.array((1.0 / 147)))
    _21 = y == np.array((0,) + (1,) + (2,))[0]
    _22 = X[_21] - _15[0, :]
    _23 = y == np.array((0,) + (1,) + (2,))[1]
    _24 = X[_23] - _15[1, :]
    _25 = y == np.array((0,) + (1,) + (2,))[2]
    _26 = X[_25] - _15[2, :]
    _27 = np.concatenate((_22,) + (_24,) + (_26,), axis=0)
    _28 = np.sum(_27, axis=0)
    _29 = _28 / np.array(_27.shape[0])
    _30 = np.expand_dims(_29, 0)
    _31 = _27 - _30
    _32 = np.square(_31)
    _33 = np.sum(_32, axis=0)
    _34 = _33 / np.array(_32.shape[0])
    _35 = np.sqrt(_34)
    _36 = _35
    _37 = _35 == np.array(0)
    _36[_37] = np.array(1.0)
    _38 = _27 / _36
    _39 = _20 * _38
    _40 = np.linalg.svd(_39, full_matrices=False)
    _41 = _40[1] > np.array(0.0001)
    _42 = np.sum(_41)
    _43 = _42.astype(np.dtype(np.int32))
    _44 = _40[2][:_43.item(), :] / _36
    _45 = _44.T / _40[1][:_43.item()]
    _46 = np.sqrt(np.array(((150 * _7.item()) * (1.0 / 2))))
    _47 = _15 - _18
    _48 = _46 * _47.T
    _49 = _48.T @ _45
    _50 = np.linalg.svd(_49, full_matrices=False)
    _51 = np.array(0.0001) * _50[1][0]
    _52 = _50[1] > _51
    _53 = np.sum(_52)
    _54 = _53.astype(np.dtype(np.int32))
    _55 = _45 @ _50[2].T[:, :_54.item()]
    _56 = _19 @ _55
    return _56[:, :2]
