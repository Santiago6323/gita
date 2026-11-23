"""
gita.audio.caracterizacion — extracción de características de audio.

Cada herramienta en su propio módulo (disvoice, y más adelante opensmile, ...).
Todas las funciones aceptan como entrada una carpeta o una lista de rutas.
"""

from .disvoice import DIMENSIONES as DIMENSIONES_DISVOICE
from .disvoice import caracterizar_disvoice

__all__ = ["DIMENSIONES_DISVOICE", "caracterizar_disvoice"]
