"""Modelos de mezcla de gaussianas (GMM) para clasificación.

- seleccionar_m: barrido del número de gaussianas M (AIC, BIC).
- graficar_seleccion_m: curva AIC/BIC vs M.
- ClasificadorGMM: un GMM por clase, decisión por máxima probabilidad a posteriori.
  Compatible con scikit-learn (Pipeline, cross_val_score, GridSearchCV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.mixture import GaussianMixture
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

__all__ = ["seleccionar_m", "graficar_seleccion_m", "ClasificadorGMM"]


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


class ClasificadorGMM(ClassifierMixin, BaseEstimator):
    """Clasificador generativo: un GMM por clase.

    Decide por argmax_c [ log p(x | c) + log P(c) ].

    Parameters
    ----------
    n_componentes : int o dict {clase: int}
        Número de gaussianas M. Un int usa el mismo M para todas las clases;
        un dict permite un M distinto por clase.
    tipo_covarianza : {"full", "tied", "diag", "spherical"}
    priors : {"empiricos", "uniformes"} o array-like
        Probabilidades a priori de cada clase (en el orden de classes_).
    n_init, max_iter, reg_covar, random_state :
        Se pasan a sklearn.mixture.GaussianMixture.

    Attributes
    ----------
    classes_ : array de clases
    modelos_ : dict {clase: GaussianMixture ajustado}
    log_priors_ : array con log P(c)
    """

    def __init__(
        self,
        n_componentes=1,
        tipo_covarianza="diag",
        priors="empiricos",
        n_init=3,
        max_iter=200,
        reg_covar=1e-6,
        random_state=None,
    ):
        self.n_componentes = n_componentes
        self.tipo_covarianza = tipo_covarianza
        self.priors = priors
        self.n_init = n_init
        self.max_iter = max_iter
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _m_de_clase(self, clase):
        if isinstance(self.n_componentes, dict):
            return self.n_componentes[clase]
        return self.n_componentes

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]

        self.modelos_ = {}
        for c in self.classes_:
            self.modelos_[c] = _crear_gmm(
                self._m_de_clase(c), self.tipo_covarianza, self.n_init,
                self.max_iter, self.reg_covar, self.random_state,
            ).fit(X[y == c])

        if isinstance(self.priors, str):
            if self.priors == "empiricos":
                p = np.array([np.mean(y == c) for c in self.classes_])
            elif self.priors == "uniformes":
                p = np.full(len(self.classes_), 1.0 / len(self.classes_))
            else:
                raise ValueError("priors debe ser 'empiricos', 'uniformes' o un array.")
        else:
            p = np.asarray(self.priors, dtype=float)
            if p.shape != (len(self.classes_),) or not np.isclose(p.sum(), 1):
                raise ValueError("priors debe tener una probabilidad por clase y sumar 1.")
        self.log_priors_ = np.log(p)
        return self

    def log_verosimilitud(self, X):
        """log p(x | c) para cada muestra y clase -> (n_muestras, n_clases)."""
        check_is_fitted(self, "modelos_")
        X = check_array(X)
        return np.column_stack(
            [self.modelos_[c].score_samples(X) for c in self.classes_]
        )

    def _log_conjunta(self, X):
        return self.log_verosimilitud(X) + self.log_priors_

    def predict_log_proba(self, X):
        lc = self._log_conjunta(X)
        return lc - logsumexp(lc, axis=1, keepdims=True)

    def predict_proba(self, X):
        return np.exp(self.predict_log_proba(X))

    def predict(self, X):
        return self.classes_[np.argmax(self._log_conjunta(X), axis=1)]

    def decision_function(self, X):
        """Binario: log-cociente a posteriori de classes_[1] vs classes_[0].
        Multiclase: log p(x|c) + log P(c) por clase."""
        lc = self._log_conjunta(X)
        if len(self.classes_) == 2:
            return lc[:, 1] - lc[:, 0]
        return lc
