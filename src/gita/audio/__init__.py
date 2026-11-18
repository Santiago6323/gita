"""
gita.audio — utilidades para señales y archivos de audio.
"""

from .preprocesamiento import (
    FS_DESTINO,
    a_mono,
    eliminar_dc,
    normalizar_pico,
    preprocesar,
    preprocesar_archivo,
    preprocesar_carpeta,
    remuestrear,
)

__all__ = [
    "FS_DESTINO",
    "a_mono",
    "eliminar_dc",
    "normalizar_pico",
    "preprocesar",
    "preprocesar_archivo",
    "preprocesar_carpeta",
    "remuestrear",
]
