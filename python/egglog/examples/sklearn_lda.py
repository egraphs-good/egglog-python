"""
Implenting an Array API to use with Scikit-learn
================================================

In this tutorial, we will create an object that implements the Array API and use it in the `LinearDiscriminantAnalysis` example that is in the [scikit-learn docs](https://scikit-learn.org/stable/modules/array_api.html).

"""
from egglog.exp.array_api import *
from sklearn import config_context, datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit(X, y):
    with config_context(array_api_dispatch=True):
        lda = LinearDiscriminantAnalysis(n_components=2)
        X_r2 = lda.fit(X, y).transform(X)
        return X_r2


iris = datasets.load_iris()

X = iris.data
y = iris.target
X_arr = NDArray.var("X")
assume_dtype(X_arr, float64)
assume_shape(X_arr, TupleInt(150) + TupleInt(4))  # type: ignore
assume_isfinite(X_arr)

y_arr = NDArray.var("y")
assume_dtype(y_arr, int64)
assume_shape(y_arr, TupleInt(150))  # type: ignore
assume_value_one_of(y_arr, (0, 1, 2))  # type: ignore

with EGraph([array_api_module]):
    res: NDArray = fit(X_arr, y_arr)
    # egraph.display()


with EGraph([array_api_module]) as egraph:
    egraph.register(res)
    egraph.run((run() * 10).saturate())
    egraph.run((run() * 10).saturate())
    res = egraph.extract(expr=res)
    print(res)
    # egraph.display()
