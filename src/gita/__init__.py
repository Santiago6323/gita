"""
gita — herramientas de procesamiento de señales y aprendizaje automático del GITA Lab (UdeA).

Los submódulos (gita.audio, gita.modelos, ...) se agregan a medida que se necesitan.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("gita-lab")
except Exception:  # paquete sin instalar (p. ej. ejecutado desde el código fuente)
    __version__ = "0.0.0"
