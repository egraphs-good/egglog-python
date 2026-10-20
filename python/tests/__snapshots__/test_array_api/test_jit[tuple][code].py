def __fn(x, y):
    _0 = 100
    assert _0 < (len(x.shape) + 2)
    return x[x.shape[_0] if (_0 < len(x.shape)) else (1, 2, )[(_0 - len(x.shape))]]
