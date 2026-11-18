#!/usr/bin/env python3
"""
Worker de DisVoice: se ejecuta con el Python del entorno de DisVoice.

NO importa gita (ese entorno no la tiene instalada): solo la librería estándar,
numpy, pandas y disvoice. Lo lanza gita.audio.caracterizacion.caracterizar_disvoice;
no está pensado para usarse directamente.

Entrada:  --lista      archivo .txt con una ruta de audio por línea
          --salida     carpeta donde escribir los resultados
          --estaticas  dimensiones a extraer en modo estático
          --dinamicas  dimensiones a extraer en modo dinámico
Salida:   <salida>/<dim>.csv      estáticas: ruta + características (head_st)
          <salida>/<dim>_dyn.csv  dinámicas: ruta + frame + características
          <salida>/errores.csv    dimension, tipo, ruta, error, traceback

Estáticas (static=True): un vector por audio. Un audio que falla queda con su
fila en NaN (no desaparece).
Dinámicas (static=False): un vector por frame o segmento. Qué es un frame
depende de la dimensión (DisVoice 0.1.10):
  - phonation   : ventana sonora
  - articulation: transición sordo->sonoro (onset); no incluye offsets ni formantes
  - prosody     : segmento sonoro
  - phonological: frame de phonet (se quita la columna 'time')
Un audio que falla no aporta frames (queda registrado en errores.csv).

En ambos modos los NaN de DisVoice se conservan y los ±inf pasan a NaN.
"""

import argparse
import os
import sys
import traceback
import warnings

# Al ejecutar un script por ruta, Python pone su carpeta primero en sys.path.
# Se quita para que ningún archivo vecino tape al paquete disvoice real.
_CARPETA = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p or ".") != _CARPETA]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")


def crear_extractor(dim):
    if dim == "phonation":
        from disvoice.phonation import Phonation
        return Phonation()
    if dim == "prosody":
        from disvoice.prosody import Prosody
        return Prosody()
    if dim == "articulation":
        from disvoice.articulation import Articulation
        return Articulation()
    if dim == "phonological":
        from disvoice.phonological import Phonological
        return Phonological()
    if dim == "glottal":
        from disvoice.glottal import Glottal
        return Glottal()
    raise ValueError("Dimensión no válida: {}".format(dim))


def nombre_corto(ruta):
    """carpeta/archivo: distingue el mismo ID en tareas distintas."""
    return os.path.join(os.path.basename(os.path.dirname(ruta)), os.path.basename(ruta))


def registrar_error(errores, dim, tipo, ruta, e):
    errores.append({
        "dimension": dim, "tipo": tipo, "ruta": ruta, "error": repr(e),
        "traceback": traceback.format_exc(),
    })
    return "  ERROR: {!r}".format(e)


def extraer_estaticas(dim, extractor, rutas, salida, errores):
    columnas = list(extractor.head_st)
    n_feat = len(columnas)

    filas = []
    n_fallos, n_inf = 0, 0
    for i, ruta in enumerate(rutas, 1):
        estado = ""
        try:
            feat = extractor.extract_features_file(ruta, static=True, plots=False, fmt="npy")
            feat = np.asarray(feat, dtype=float).ravel()
            if feat.size != n_feat:
                raise ValueError("{} características, se esperaban {}".format(feat.size, n_feat))
            mascara_inf = np.isinf(feat)
            if mascara_inf.any():
                n_inf += int(mascara_inf.sum())
                feat[mascara_inf] = np.nan
        except Exception as e:
            n_fallos += 1
            feat = np.full(n_feat, np.nan)
            estado = registrar_error(errores, dim, "estaticas", ruta, e)
        filas.append(feat)
        print("[{} estáticas] {}/{} {}{}".format(dim, i, len(rutas), nombre_corto(ruta), estado),
              flush=True)

    df = pd.DataFrame(np.vstack(filas), columns=columnas)
    df.insert(0, "ruta", rutas)
    df.to_csv(os.path.join(salida, dim + ".csv"), index=False)

    todo_nan = int(df[columnas].isna().all(axis=1).sum())
    print("[{} estáticas] listo: {} audios x {} características | fallos: {} | filas todo-NaN: {} | "
          "inf->NaN: {}".format(dim, len(df), n_feat, n_fallos, todo_nan, n_inf), flush=True)


def a_dataframe_dinamico(feat, extractor):
    """Convierte la salida dinámica de DisVoice en un DataFrame numérico (frames x caract.)."""
    if isinstance(feat, pd.DataFrame):
        df = feat.reset_index(drop=True).copy()
    else:
        arr = np.asarray(feat, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        head = list(getattr(extractor, "head_dyn", []) or [])
        cols = head if len(head) == arr.shape[1] else ["f{}".format(k) for k in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=cols)
    df = df.drop(columns=[c for c in df.columns if str(c).lower() == "time"], errors="ignore")
    return df.apply(pd.to_numeric, errors="coerce")


def extraer_dinamicas(dim, extractor, rutas, salida, errores):
    partes, frames = [], {}
    columnas = None
    n_fallos, n_inf = 0, 0
    for i, ruta in enumerate(rutas, 1):
        estado = ""
        try:
            feat = extractor.extract_features_file(ruta, static=False, plots=False, fmt="dataframe")
            df = a_dataframe_dinamico(feat, extractor)
            if columnas is None:
                columnas = list(df.columns)
            elif list(df.columns) != columnas:
                raise ValueError("columnas distintas a las del primer audio")
            mascara_inf = np.isinf(df.to_numpy(dtype=float))
            if mascara_inf.any():
                n_inf += int(mascara_inf.sum())
                df = df.mask(mascara_inf)
            df.insert(0, "frame", np.arange(len(df)))
            df.insert(0, "ruta", ruta)
            partes.append(df)
            frames[ruta] = len(df)
            estado = "  ({} frames)".format(len(df))
        except Exception as e:
            n_fallos += 1
            estado = registrar_error(errores, dim, "dinamicas", ruta, e)
        print("[{} dinámicas] {}/{} {}{}".format(dim, i, len(rutas), nombre_corto(ruta), estado),
              flush=True)

    if partes:
        df = pd.concat(partes, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["ruta", "frame"])
    df.to_csv(os.path.join(salida, dim + "_dyn.csv"), index=False)

    if frames:
        n = np.array(list(frames.values()))
        print("[{} dinámicas] listo: {} frames de {} audios x {} características | frames por audio: "
              "min={} mediana={} max={} | fallos: {} | inf->NaN: {}".format(
                  dim, len(df), len(frames), len(columnas), n.min(), int(np.median(n)), n.max(),
                  n_fallos, n_inf), flush=True)
        pocos = [nombre_corto(r) for r, k in frames.items() if k < 20]
        if pocos:
            print("[{} dinámicas] AVISO: {} audios con menos de 20 frames: {}".format(
                dim, len(pocos), pocos[:10]), flush=True)
    else:
        print("[{} dinámicas] ningún audio produjo frames | fallos: {}".format(dim, n_fallos),
              flush=True)


def main():
    parser = argparse.ArgumentParser(description="Worker de DisVoice para gita")
    parser.add_argument("--lista", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--estaticas", nargs="*", default=[])
    parser.add_argument("--dinamicas", nargs="*", default=[])
    args = parser.parse_args()

    with open(args.lista, encoding="utf-8") as f:
        rutas = [linea.strip() for linea in f if linea.strip()]

    errores = []
    # Cada extractor se crea una sola vez y sirve para ambos modos
    for dim in dict.fromkeys(args.estaticas + args.dinamicas):
        extractor = crear_extractor(dim)
        if dim in args.estaticas:
            extraer_estaticas(dim, extractor, rutas, args.salida, errores)
        if dim in args.dinamicas:
            extraer_dinamicas(dim, extractor, rutas, args.salida, errores)

    pd.DataFrame(errores, columns=["dimension", "tipo", "ruta", "error", "traceback"]).to_csv(
        os.path.join(args.salida, "errores.csv"), index=False
    )


if __name__ == "__main__":
    main()
