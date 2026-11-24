import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from gita.modelos.gmm import ClasificadorGMM, graficar_seleccion_m, seleccionar_m


@pytest.fixture
def datos_3_grupos():
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=0)
    return X


@pytest.fixture
def datos_binarios():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 1, (150, 4))
    X1 = rng.normal(2, 1, (150, 4))
    return np.vstack([X0, X1]), np.array([0] * 150 + [1] * 150)


def test_seleccionar_m_columnas_y_bic(datos_3_grupos):
    res = seleccionar_m(datos_3_grupos, m_valores=range(1, 7), random_state=0)
    assert list(res.columns) == ["m", "n_parametros", "aic", "bic", "convergio"]
    assert list(res["m"]) == list(range(1, 7))
    assert res.loc[res["bic"].idxmin(), "m"] == 3


def test_n_parametros_coincide_con_sklearn():
    from sklearn.mixture import GaussianMixture
    from gita.modelos.gmm import _n_parametros
    X = np.random.default_rng(0).normal(size=(200, 5))
    for tipo in ("full", "tied", "diag", "spherical"):
        g = GaussianMixture(3, covariance_type=tipo, random_state=0).fit(X)
        assert g._n_parameters() == _n_parametros(3, 5, tipo)


def test_graficar(datos_3_grupos):
    import matplotlib
    matplotlib.use("Agg")
    res = seleccionar_m(datos_3_grupos, m_valores=range(1, 4), random_state=0)
    ax = graficar_seleccion_m(res, titulo="prueba")
    assert len(ax.lines) == 4  # AIC, BIC y sus dos mínimos


def test_clasificador_basico(datos_binarios):
    X, y = datos_binarios
    clf = ClasificadorGMM(n_componentes=2, random_state=0).fit(X, y)
    assert clf.score(X, y) > 0.9
    proba = clf.predict_proba(X)
    assert proba.shape == (300, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1)
    assert clf.decision_function(X).shape == (300,)


def test_m_por_clase_y_priors(datos_binarios):
    X, y = datos_binarios
    clf = ClasificadorGMM(n_componentes={0: 1, 1: 3}, priors=[0.3, 0.7],
                          random_state=0).fit(X, y)
    assert clf.modelos_[0].n_components == 1
    assert clf.modelos_[1].n_components == 3
    np.testing.assert_allclose(np.exp(clf.log_priors_), [0.3, 0.7])


def test_priors_invalidos(datos_binarios):
    X, y = datos_binarios
    with pytest.raises(ValueError):
        ClasificadorGMM(priors=[0.5, 0.6]).fit(X, y)


def test_compatible_sklearn(datos_binarios):
    X, y = datos_binarios
    pipe = make_pipeline(StandardScaler(), ClasificadorGMM(n_componentes=2, random_state=0))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
    assert scores.mean() > 0.9
    assert clone(ClasificadorGMM(n_componentes=4)).n_componentes == 4
