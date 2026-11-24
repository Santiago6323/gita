"""Modelos de mezcla de gaussianas (GMM) para clasificación.

- seleccionar_m: barrido del número de gaussianas M (AIC, BIC).
- graficar_seleccion_m: curva AIC/BIC vs M.
- momentos_gmm: media y covarianza globales de un GMM.
- distancia_mahalanobis_gmm, distancia_bhattacharyya_gmm: distancias entre dos
  GMMs ajustados de forma independiente (pueden tener distinto M).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_array, check_is_fitted

__all__ = [
    "seleccionar_m",
    "graficar_seleccion_m",
    "momentos_gmm",
    "distancia_mahalanobis_gmm",
    "distancia_bhattacharyya_gmm",
]


def _crear_gmm(m, tipo_covarianza, n_init, max_iter, reg_covar, random_state):
    return GaussianMixture(
        n_components=m,
        covariance_type=tipo_covarianza,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=reg_covar,
        random_state=random_state,
    )


def _n_parametros(m, d, tipo_covarianza):
    """Parámetros libres del GMM: medias + covarianzas + (M - 1) pesos."""
    cov = {
        "full": m * d * (d + 1) // 2,
        "tied": d * (d + 1) // 2,
        "diag": m * d,
        "spherical": m,
    }[tipo_covarianza]
    return m * d + cov + (m - 1)


def seleccionar_m(
    X,
    m_valores=range(1, 11),
    tipo_covarianza="diag",
    n_init=3,
    max_iter=200,
    reg_covar=1e-6,
    random_state=None,
):
    """Ajusta un GMM para cada M y calcula AIC y BIC.

    AIC = -2 ln(L) + 2k
    BIC = -2 ln(L) + k ln(n)
    con L la verosimilitud máxima, k el número de parámetros libres y
    n el número de muestras. El M adecuado es el que minimiza el criterio.

    Parameters
    ----------
    X : array (n_muestras, n_caracteristicas)
        Matriz de características (idealmente ya estandarizada).
    m_valores : iterable de int
        Números de gaussianas a evaluar.
    tipo_covarianza : {"full", "tied", "diag", "spherical"}
    n_init, max_iter, reg_covar, random_state :
        Se pasan a sklearn.mixture.GaussianMixture.

    Returns
    -------
    pandas.DataFrame con columnas: m, n_parametros, aic, bic, convergio
    """
    X = check_array(X)
    filas = []
    for m in m_valores:
        gmm = _crear_gmm(m, tipo_covarianza, n_init, max_iter, reg_covar,
                         random_state).fit(X)
        filas.append({
            "m": m,
            "n_parametros": _n_parametros(m, X.shape[1], tipo_covarianza),
            "aic": gmm.aic(X),
            "bic": gmm.bic(X),
            "convergio": gmm.converged_,
        })
    return pd.DataFrame(filas)


def graficar_seleccion_m(resultados, titulo=None, ax=None):
    """Grafica AIC y BIC vs número de gaussianas (resultado de seleccionar_m).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4.5))
    m = resultados["m"]
    ax.plot(m, resultados["aic"], "b-", label="AIC")
    ax.plot(m, resultados["bic"], "g--", label="BIC")
    for col, color in (("aic", "b"), ("bic", "g")):
        i = resultados[col].idxmin()
        ax.plot(m[i], resultados[col][i], "o", color=color, markersize=8)
    ax.set_xlabel("Número de componentes (M)")
    ax.set_ylabel("Criterio de información")
    ax.set_xticks(list(m))
    ax.legend()
    ax.grid(alpha=0.3)
    if titulo:
        ax.set_title(titulo)
    return ax


# --- distancias entre GMMs --------------------------------------------------

def _covarianzas_completas(gmm):
    """Covarianzas como arreglo (M, d, d) para cualquier covariance_type."""
    m, d = gmm.means_.shape
    c = gmm.covariances_
    tipo = gmm.covariance_type
    if tipo == "full":
        return c
    if tipo == "diag":
        return np.stack([np.diag(v) for v in c])
    if tipo == "spherical":
        return np.stack([np.eye(d) * v for v in c])
    if tipo == "tied":
        return np.repeat(c[None], m, axis=0)
    raise ValueError(f"covariance_type desconocido: {tipo}")


def _verificar(gmm_a, gmm_b):
    for g in (gmm_a, gmm_b):
        check_is_fitted(g, "means_")
    if gmm_a.means_.shape[1] != gmm_b.means_.shape[1]:
        raise ValueError(f"Los GMMs tienen distinto número de características "
                         f"({gmm_a.means_.shape[1]} vs {gmm_b.means_.shape[1]}).")


def momentos_gmm(gmm):
    """Media y covarianza globales de un GMM (la gaussiana equivalente).

        mu    = sum_i w_i mu_i
        Sigma = sum_i w_i (Sigma_i + mu_i mu_i') - mu mu'
    """
    check_is_fitted(gmm, "means_")
    w, mus = gmm.weights_, gmm.means_
    mu = w @ mus
    covs = _covarianzas_completas(gmm)
    segundo = np.einsum("i,ijk->jk", w, covs + np.einsum("ij,ik->ijk", mus, mus))
    return mu, segundo - np.outer(mu, mu)


def distancia_mahalanobis_gmm(gmm_a, gmm_b):
    """Distancia de Mahalanobis entre dos GMMs.

    Cada GMM se resume en su gaussiana equivalente (momentos_gmm) y:
        D_M = sqrt(delta' S^-1 delta),  delta = mu_a - mu_b,  S = (Sigma_a + Sigma_b) / 2
    Mide qué tan separados están los centros de las dos distribuciones en
    unidades de su dispersión; no ve diferencias de forma con la misma media.
    """
    _verificar(gmm_a, gmm_b)
    mu_a, cov_a = momentos_gmm(gmm_a)
    mu_b, cov_b = momentos_gmm(gmm_b)
    delta = mu_a - mu_b
    s = (cov_a + cov_b) / 2
    return float(np.sqrt(delta @ np.linalg.solve(s, delta)))


def distancia_bhattacharyya_gmm(gmm_a, gmm_b, n_muestras=20000, random_state=None):
    """Distancia de Bhattacharyya entre dos GMMs (estimada por Monte Carlo).

        BC  = integral sqrt(p_a(x) p_b(x)) dx      (coeficiente, entre 0 y 1)
        D_B = -ln(BC)                              (0 si son idénticos)

    Entre mezclas no hay fórmula cerrada. Se estima muestreando de la mezcla
    m = (p_a + p_b) / 2:  BC = E_m[ sqrt(p_a p_b) / m ].  Como
    sqrt(p_a p_b) <= m, cada término está entre 0 y 1 y el estimador es
    estable. Compara las distribuciones completas (posición y forma) y
    funciona aunque los GMMs tengan distinto número de gaussianas.

    Parameters
    ----------
    gmm_a, gmm_b : GaussianMixture ajustados (mismo número de características)
    n_muestras : int
        Muestras de Monte Carlo (la mitad de cada GMM). Más muestras -> menos
        varianza en la estimación.
    random_state : int o None
    """
    _verificar(gmm_a, gmm_b)
    n_a = n_muestras // 2
    x = np.vstack([
        _muestrear(gmm_a, n_a, random_state),
        _muestrear(gmm_b, n_muestras - n_a, None if random_state is None else random_state + 1),
    ])
    lp_a, lp_b = gmm_a.score_samples(x), gmm_b.score_samples(x)
    # ln[ sqrt(p_a p_b) / ((p_a + p_b) / 2) ]  (<= 0)
    log_r = 0.5 * (lp_a + lp_b) - np.logaddexp(lp_a, lp_b) + np.log(2)
    log_bc = logsumexp(log_r) - np.log(len(x))
    return float(max(0.0, -log_bc))


def _muestrear(gmm, n, random_state):
    """Muestras de un GMM sin modificar su random_state interno."""
    rng = np.random.default_rng(random_state)
    comp = rng.choice(len(gmm.weights_), size=n, p=gmm.weights_)
    covs = _covarianzas_completas(gmm)
    x = np.empty((n, gmm.means_.shape[1]))
    for i in np.unique(comp):
        sel = comp == i
        x[sel] = rng.multivariate_normal(gmm.means_[i], covs[i], size=sel.sum())
    return x
