import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from gita.modelos.gmm import (
    distancia_bhattacharyya_gmm,
    distancia_mahalanobis_gmm,
    graficar_seleccion_m,
    momentos_gmm,
    seleccionar_m,
)


@pytest.fixture
def datos_3_grupos():
    X, _ = make_blobs(n_samples=600, centers=3, cluster_std=0.6, random_state=0)
    return X


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


# --- distancias entre GMMs --------------------------------------------------

def _gaussiana(mu, cov):
    """GMM de una sola gaussiana con parámetros conocidos."""
    mu = np.atleast_1d(np.asarray(mu, float))
    cov = np.atleast_2d(np.asarray(cov, float))
    g = GaussianMixture(1, covariance_type="full")
    g.weights_ = np.array([1.0])
    g.means_ = mu[None]
    g.covariances_ = cov[None]
    g.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(cov))[None]
    return g


@pytest.fixture
def gmms():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (2000, 4)), rng.normal(4, 1, (2000, 4))])
    g2 = GaussianMixture(2, covariance_type="diag", random_state=0).fit(X)
    g5 = GaussianMixture(5, covariance_type="diag", random_state=0).fit(X)       # mismos datos, otro M
    g_desp = GaussianMixture(3, covariance_type="diag", random_state=0).fit(X + 1)
    return g2, g5, g_desp


def test_momentos_gmm():
    g = GaussianMixture(2, covariance_type="diag")
    g.weights_ = np.array([0.5, 0.5])
    g.means_ = np.array([[0.0], [2.0]])
    g.covariances_ = np.array([[1.0], [1.0]])
    mu, cov = momentos_gmm(g)
    np.testing.assert_allclose(mu, [1.0])
    np.testing.assert_allclose(cov, [[2.0]])          # 1 (dentro) + 1 (entre medias)


def test_distancias_casos_cerrados():
    a, b = _gaussiana([0], [[1]]), _gaussiana([1], [[1]])
    assert distancia_mahalanobis_gmm(a, b) == pytest.approx(1.0)
    assert distancia_bhattacharyya_gmm(a, b, 100000, random_state=0) == pytest.approx(0.125, abs=3e-3)

    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 5))
    sa, sb = A @ A.T + np.eye(5), np.diag(rng.uniform(0.5, 3, 5))
    ma, mb = np.zeros(5), rng.normal(size=5)
    s, d = (sa + sb) / 2, ma - mb
    exacto = d @ np.linalg.solve(s, d) / 8 + 0.5 * (
        np.linalg.slogdet(s)[1] - 0.5 * (np.linalg.slogdet(sa)[1] + np.linalg.slogdet(sb)[1]))
    estimado = distancia_bhattacharyya_gmm(_gaussiana(ma, sa), _gaussiana(mb, sb), 100000, random_state=0)
    assert estimado == pytest.approx(exacto, rel=0.02)


def test_distancias_identicos_y_distinto_m(gmms):
    g2, g5, g_desp = gmms
    assert distancia_bhattacharyya_gmm(g2, g2, random_state=0) == 0.0
    assert distancia_mahalanobis_gmm(g2, g2) == pytest.approx(0.0, abs=1e-10)
    # misma distribución con distinto M: casi 0; desplazada: claramente mayor
    assert distancia_bhattacharyya_gmm(g2, g5, random_state=0) < 0.02
    assert distancia_bhattacharyya_gmm(g2, g_desp, random_state=0) > 0.3
    assert distancia_mahalanobis_gmm(g2, g_desp) > 0.3


def test_distancias_simetricas_y_reproducibles(gmms):
    g2, _, g_desp = gmms
    assert distancia_mahalanobis_gmm(g2, g_desp) == pytest.approx(distancia_mahalanobis_gmm(g_desp, g2))
    ab = distancia_bhattacharyya_gmm(g2, g_desp, random_state=1)
    ba = distancia_bhattacharyya_gmm(g_desp, g2, random_state=1)
    assert ab == pytest.approx(ba, abs=0.02)
    assert ab == distancia_bhattacharyya_gmm(g2, g_desp, random_state=1)


def test_distancias_dimensiones_distintas(gmms):
    g2 = gmms[0]
    with pytest.raises(ValueError):
        distancia_mahalanobis_gmm(g2, _gaussiana([0, 0], np.eye(2)))
