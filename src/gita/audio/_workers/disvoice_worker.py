#!/usr/bin/env python3
"""
Worker de DisVoice: se ejecuta con el Python del entorno de DisVoice.

NO importa gita (ese entorno no la tiene instalada): solo la librería estándar,
numpy, pandas y disvoice. Lo lanza gita.audio.disvoice.caracterizar_disvoice;
no está pensado para usarse directamente.

Entrada:  --lista  archivo .txt con una ruta de audio por línea
          --salida carpeta donde escribir los resultados
          --dims   dimensiones a extraer (phonation, prosody, ...)
Salida:   <salida>/<dim>.csv   columnas: ruta + características (head_st)
          <salida>/errores.csv columnas: dimension, ruta, error, traceback

Características estáticas (static=True): un vector por audio.
- Un audio que falla queda con su fila en NaN (no desaparece) y su error en
  errores.csv.
- Los NaN que entrega DisVoice se dejan como NaN; los ±inf se convierten a NaN.
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


def extraer_estaticas(dim, rutas, salida, errores):
    extractor = crear_extractor(dim)
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
            errores.append({
                "dimension": dim, "ruta": ruta, "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            estado = "  ERROR: {!r}".format(e)
        filas.append(feat)
        print("[{}] {}/{} {}{}".format(dim, i, len(rutas), os.path.basename(ruta), estado), flush=True)

    df = pd.DataFrame(np.vstack(filas), columns=columnas)
    df.insert(0, "ruta", rutas)
    df.to_csv(os.path.join(salida, dim + ".csv"), index=False)

    caract = df.drop(columns="ruta")
    todo_nan = int(caract.isna().all(axis=1).sum())
    print("[{}] listo: {} audios x {} características | fallos: {} | filas todo-NaN: {} | "
          "inf->NaN: {}".format(dim, len(df), n_feat, n_fallos, todo_nan, n_inf), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Worker de DisVoice para gita")
    parser.add_argument("--lista", required=True)
    parser.add_argument("--salida", required=True)
    parser.add_argument("--dims", nargs="+", required=True)
    args = parser.parse_args()

    with open(args.lista, encoding="utf-8") as f:
        rutas = [linea.strip() for linea in f if linea.strip()]

    errores = []
    for dim in args.dims:
        extraer_estaticas(dim, rutas, args.salida, errores)

    pd.DataFrame(errores, columns=["dimension", "ruta", "error", "traceback"]).to_csv(
        os.path.join(args.salida, "errores.csv"), index=False
    )


if __name__ == "__main__":
    main()
