def __fn(X, i, j):
    _0 = X[(0, 0, i, j, )]
    _1 = X[(0, 1, i, j, )]
    _2 = X[(1, 0, i, j, )]
    _3 = X[(1, 1, i, j, )]
    _4 = X[(2, 0, i, j, )]
    _5 = X[(2, 1, i, j, )]
    return np.sqrt((((((np.real((np.conj(_0) * _0)) + np.real((np.conj(_1) * _1))) + np.real((np.conj(_2) * _2))) + np.real((np.conj(_3) * _3))) + np.real((np.conj(_4) * _4))) + np.real((np.conj(_5) * _5))))
