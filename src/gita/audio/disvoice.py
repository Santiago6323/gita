"""
Caracterización con DisVoice (https://github.com/jcvasquezc/DisVoice).

DisVoice necesita su propio entorno de conda (TensorFlow, phonet, ... con
versiones fijas), así que gita NO lo importa: lanza _workers/disvoice_worker.py como
un proceso aparte con el Python de ese entorno y lee sus resultados.

    entorno de gita                         entorno de DisVoice
    caracterizar_disvoice(entrada)
      lista de audios -> .txt temporal
      subprocess ---------------------->  _workers/disvoice_worker.py
                                            extrae y escribe <dim>.csv
      lee los CSV  <---------------------
      devuelve un DataFrame por dimensión

El Python de DisVoice se indica con python_disvoice=... o con la variable de
entorno GITA_DISVOICE_PYTHON.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from ..datos import listar_archivos

DIMENSIONES = ("phonation", "prosody", "articulation", "phonological", "glottal")
WORKER = Path(__file__).parent / "_workers" / "disvoice_worker.py"


def _python_disvoice(python_disvoice):
    ruta = python_disvoice or os.environ.get("GITA_DISVOICE_PYTHON")
    if not ruta:
        raise ValueError(
            "No se sabe qué Python usar para DisVoice. Pasa python_disvoice=... o define "
            "la variable de entorno GITA_DISVOICE_PYTHON (p. ej. "
            "~/miniconda3/envs/disvoice/bin/python)."
        )
    ruta = Path(ruta).expanduser()
    if not ruta.is_file():
        raise FileNotFoundError("No existe el Python de DisVoice: {}".format(ruta))
    return str(ruta)


def _ejecutar_worker(python, rutas, dimensiones, carpeta_tmp, verbose):
    lista = Path(carpeta_tmp) / "lista.txt"
    lista.write_text("\n".join(rutas) + "\n", encoding="utf-8")

    comando = [python, str(WORKER), "--lista", str(lista), "--salida", str(carpeta_tmp),
               "--dims"] + list(dimensiones)
    entorno = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONWARNINGS="ignore")

    # Se lee la salida línea a línea para mostrar el progreso (también en Spyder)
    proceso = subprocess.Popen(
        comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=entorno,
    )
    lineas = []
    for linea in proceso.stdout:
        lineas.append(linea)
        if verbose:
            print(linea, end="")
    codigo = proceso.wait()
    if codigo != 0:
        raise RuntimeError(
            "El proceso de DisVoice terminó con error (código {}). Últimas líneas:\n{}".format(
                codigo, "".join(lineas[-20:]))
        )


def caracterizar_disvoice(
    entrada,
    dimensiones=("phonation",),
    python_disvoice=None,
    carpeta_salida=None,
    sobrescribir=False,
    verbose=True,
):
    """Extrae características estáticas de DisVoice (un vector por audio).

    - entrada: carpeta de audios (.wav, incluye subcarpetas) o lista de rutas
      (p. ej. seleccionar_datos(...)["seleccionados"]).
    - dimensiones: una o varias de DIMENSIONES ("phonation", "prosody",
      "articulation", "phonological", "glottal").
    - python_disvoice: Python del entorno de DisVoice. Si es None, se usa la
      variable de entorno GITA_DISVOICE_PYTHON.
    - carpeta_salida: si se da, guarda <dimension>.csv y errores_disvoice.csv.
      Una dimensión cuyo CSV ya existe no se vuelve a extraer (se carga), a menos
      que sobrescribir=True. Si el CSV existente es de otra lista de audios,
      lanza ValueError.
    - phonological exige audios a 16 kHz (phonet devuelve NaN con otras
      frecuencias): si alguno no lo está, se lanza ValueError antes de extraer.

    Cada DataFrame tiene las columnas id (nombre del archivo sin extensión),
    ruta (ruta absoluta; distingue el mismo id en tareas distintas) y las
    características de DisVoice. Un audio que falla queda con su fila en NaN.

    Devuelve un dict con:
    - "caracteristicas": {dimension: DataFrame}
    - "fallidos": lista de dicts con dimension, ruta y error
    """
    if isinstance(dimensiones, str):
        dimensiones = [dimensiones]
    invalidas = [d for d in dimensiones if d not in DIMENSIONES]
    if invalidas:
        raise ValueError("Dimensiones no válidas: {}. Opciones: {}".format(invalidas, DIMENSIONES))

    rutas = [str(p.resolve()) for p in listar_archivos(entrada, extensiones=(".wav",))]
    if not rutas:
        raise ValueError("No hay archivos .wav en {}".format(entrada))

    caracteristicas = {}
    fallidos = []
    pendientes = list(dimensiones)

    # 1. Reutilizar los CSV que ya existan (si son de la misma lista de audios)
    if carpeta_salida is not None:
        carpeta_salida = Path(carpeta_salida)
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        for dim in dimensiones:
            csv = carpeta_salida / (dim + ".csv")
            if csv.exists() and not sobrescribir:
                df = pd.read_csv(csv)
                if set(df["ruta"]) != set(rutas):
                    raise ValueError(
                        "{} es de otra lista de audios. Usa sobrescribir=True o otra "
                        "carpeta_salida.".format(csv))
                caracteristicas[dim] = df
                pendientes.remove(dim)
                if verbose:
                    print("[{}] ya existe {}: se carga sin volver a extraer".format(dim, csv))

    # 2. phonological necesita 16 kHz: verificar antes de lanzar el proceso
    if "phonological" in pendientes:
        no_16k = [r for r in rutas if sf.info(r).samplerate != 16000]
        if no_16k:
            raise ValueError(
                "phonological requiere audios a 16 kHz (con otra frecuencia phonet devuelve "
                "NaN). {} audios no lo están, p. ej. {}. Preprocésalos antes con "
                "gita.audio.preprocesar_carpeta.".format(len(no_16k), no_16k[0]))

    # 3. Extraer las dimensiones pendientes en un solo proceso
    if pendientes:
        python = _python_disvoice(python_disvoice)
        with tempfile.TemporaryDirectory() as tmp:
            _ejecutar_worker(python, rutas, pendientes, tmp, verbose)

            for dim in pendientes:
                df = pd.read_csv(Path(tmp) / (dim + ".csv"))
                df.insert(0, "id", [Path(r).stem for r in df["ruta"]])
                caracteristicas[dim] = df
                if carpeta_salida is not None:
                    df.to_csv(carpeta_salida / (dim + ".csv"), index=False)

            errores = pd.read_csv(Path(tmp) / "errores.csv")
            if carpeta_salida is not None and len(errores):
                errores.to_csv(carpeta_salida / "errores_disvoice.csv", index=False)
            fallidos = errores[["dimension", "ruta", "error"]].to_dict("records")

    return {
        "caracteristicas": {d: caracteristicas[d] for d in dimensiones},
        "fallidos": fallidos,
    }
