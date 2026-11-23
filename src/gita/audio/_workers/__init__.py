"""
Scripts que se ejecutan con el Python de otros entornos (p. ej. el de DisVoice).

No se importan desde gita: se lanzan por ruta con subprocess. Están en su
propia carpeta para que ningún módulo de gita (como gita/audio/disvoice.py)
quede en sus rutas de búsqueda y tape al paquete real.
"""
