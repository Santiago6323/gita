"""
gita.modelos — modelos de mezcla de gaussianas (GMM): selección de M y distancias.
"""

from .gmm import (
    distancia_bhattacharyya_gmm,
    distancia_mahalanobis_gmm,
    graficar_seleccion_m,
    momentos_gmm,
    seleccionar_m,
)

__all__ = [
    "distancia_bhattacharyya_gmm",
    "distancia_mahalanobis_gmm",
    "graficar_seleccion_m",
    "momentos_gmm",
    "seleccionar_m",
]
