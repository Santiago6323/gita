"""
Remuestreo y normalización de señales de audio.

Flujo estándar (``preprocesar``):

    señal -> mono -> remuestreo a fs_destino -> normalización por pico

El orden importa: se normaliza al final porque el filtro antialiasing del
remuestreo puede cambiar el valor pico de la señal.

Convención de forma: una señal es un arreglo 1D (n,) si es mono o 2D
(n, canales) si tiene varios canales, igual que lo devuelve ``soundfile.read``.
"""

import os
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FS_DESTINO = 16000


def a_mono(senal):
    """Convierte una señal (n, canales) a mono (n,) promediando los canales.

    Si la señal ya es 1D, la devuelve sin cambios (como float64).
    """
    senal = np.asarray(senal, dtype=np.float64)
    if senal.ndim == 1:
        return senal
    if senal.ndim == 2:
        return senal.mean(axis=1)
    raise ValueError(
        "La señal debe ser 1D (n,) o 2D (n, canales); tiene forma {}".format(senal.shape)
    )


def remuestrear(senal, fs_origen, fs_destino=FS_DESTINO):
    """Remuestrea la señal de fs_origen a fs_destino (Hz).

    Usa remuestreo polifásico (scipy.signal.resample_poly) con factor racional
    up/down = fs_destino/fs_origen simplificado por el máximo común divisor.
    Incluye filtro antialiasing. Funciona con señales 1D o (n, canales).
    """
    fs_origen = int(fs_origen)
    fs_destino = int(fs_destino)
    if fs_origen <= 0 or fs_destino <= 0:
        raise ValueError("Las frecuencias de muestreo deben ser positivas")

    senal = np.asarray(senal, dtype=np.float64)
    if fs_origen == fs_destino:
        return senal.copy()

    divisor = gcd(fs_origen, fs_destino)
    up = fs_destino // divisor
    down = fs_origen // divisor
    return resample_poly(senal, up, down, axis=0)


def normalizar_pico(senal, pico=1.0):
    """Escala la señal para que max(|senal|) = pico.

    Una señal en silencio absoluto (todo ceros) se devuelve sin cambios.
    """
    if pico <= 0:
        raise ValueError("pico debe ser positivo")
    senal = np.asarray(senal, dtype=np.float64)
    maximo = np.max(np.abs(senal)) if senal.size else 0.0
    if maximo == 0:
        return senal.copy()
    return senal * (pico / maximo)


def preprocesar(senal, fs, fs_destino=FS_DESTINO, mono=True, normalizar=True, pico=1.0):
    """Aplica mono -> remuestreo -> normalización por pico.

    Devuelve (senal_procesada, fs_destino).
    """
    if mono:
        senal = a_mono(senal)
    senal = remuestrear(senal, fs, fs_destino)
    if normalizar:
        senal = normalizar_pico(senal, pico)
    return senal, int(fs_destino)


def preprocesar_archivo(
    ruta_entrada,
    ruta_salida,
    fs_destino=FS_DESTINO,
    mono=True,
    normalizar=True,
    pico=1.0,
    subtipo="PCM_16",
):
    """Lee un archivo de audio, lo preprocesa y lo guarda en ruta_salida.

    - Crea las carpetas de ruta_salida si no existen.
    - subtipo: formato de muestra del archivo de salida ("PCM_16", "PCM_24",
      "FLOAT", ...). None deja el predeterminado de soundfile.

    Devuelve (ruta_salida, fs_destino).
    """
    senal, fs = sf.read(ruta_entrada, dtype="float64", always_2d=False)
    senal, fs = preprocesar(senal, fs, fs_destino, mono, normalizar, pico)

    carpeta = os.path.dirname(os.path.abspath(ruta_salida))
    os.makedirs(carpeta, exist_ok=True)
    sf.write(ruta_salida, senal, fs, subtype=subtipo)
    return ruta_salida, fs
