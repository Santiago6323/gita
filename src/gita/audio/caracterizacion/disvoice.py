"""
Caracterización con DisVoice (https://github.com/jcvasquezc/DisVoice).

DisVoice necesita su propio entorno de conda (TensorFlow, phonet, ... con
versiones fijas), así que gita NO lo importa: lanza _workers/disvoice_worker.py
como un proceso aparte con el Python de ese entorno y lee sus resultados.

    entorno de gita                         entorno de DisVoice
    caracterizar_disvoice(entrada)
      lista de audios -> .txt temporal
      subprocess ---------------------->  _workers/disvoice_worker.py
                                            extrae y escribe un CSV por dimensión
      lee los CSV  <---------------------
      devuelve DataFrames y los guarda en carpeta_salida

Python de DisVoice, en este orden:
  1. el parámetro python_disvoice
  2. la variable de entorno GITA_DISVOICE_PYTHON
  3. un entorno de conda llamado "disvoice" junto al entorno actual
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import soundfile as sf

from ...caracteristicas import ESTRATEGIAS, guardar_pkl, limpiar_nan
from ...datos import listar_archivos

DIMENSIONES = ("phonation", "prosody", "articulation", "phonological", "glottal")
TIPOS = ("estaticas", "dinamicas")
_ETIQUETA = {"estaticas": "estáticas", "dinamicas": "dinámicas"}
WORKER = Path(__file__).resolve().parents[1] / "_workers" / "disvoice_worker.py"
ARCHIVO_ERRORES = "errores_disvoice.csv"
COLUMNAS_ERRORES = ["dimension", "tipo", "ruta", "error", "traceback"]

# Líneas de la salida del worker que no aportan: barras de progreso de Keras
# (phonet), el aviso de KALDI_ROOT y líneas vacías.
_RUIDO = re.compile(r"^\s*(\d+/\d+ \[|#{5,}|###\s|$)")


def _python_disvoice(python_disvoice):
    if python_disvoice:
        candidatos = [("python_disvoice", Path(python_disvoice).expanduser())]
    elif os.environ.get("GITA_DISVOICE_PYTHON"):
        candidatos = [("GITA_DISVOICE_PYTHON", Path(os.environ["GITA_DISVOICE_PYTHON"]).expanduser())]
    else:
        # Entorno de conda "disvoice" hermano del actual (envs/portables -> envs/disvoice)
        # o dentro de la instalación base (miniconda3 -> miniconda3/envs/disvoice)
        ejecutable = Path("python.exe") if os.name == "nt" else Path("bin") / "python"
        prefijo = Path(sys.prefix)
        candidatos = [
            ("entorno conda 'disvoice'", prefijo.parent / "disvoice" / ejecutable),
            ("entorno conda 'disvoice'", prefijo / "envs" / "disvoice" / ejecutable),
        ]

    for _, ruta in candidatos:
        if ruta.is_file():
            return str(ruta)

    origen = candidatos[0][0]
    if origen == "entorno conda 'disvoice'":
        raise ValueError(
            "No se encontró el Python de DisVoice (se buscó en {}). Pasa python_disvoice=... "
            "o define la variable de entorno GITA_DISVOICE_PYTHON.".format(
                ", ".join(str(r) for _, r in candidatos)))
    raise FileNotFoundError("No existe el Python de DisVoice ({}): {}".format(origen, candidatos[0][1]))


def _nombre_csv(dim, tipo):
    return dim + ("_dyn" if tipo == "dinamicas" else "") + ".csv"


def _ejecutar_worker(python, rutas, pendientes, carpeta_tmp, verbose):
    lista = Path(carpeta_tmp) / "lista.txt"
    lista.write_text("\n".join(rutas) + "\n", encoding="utf-8")

    comando = [python, str(WORKER), "--lista", str(lista), "--salida", str(carpeta_tmp)]
    for tipo in TIPOS:
        if pendientes[tipo]:
            comando += ["--" + tipo] + pendientes[tipo]
    entorno = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", PYTHONWARNINGS="ignore")

    # Se lee la salida línea a línea para mostrar el progreso (también en Spyder)
    proceso = subprocess.Popen(
        comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True, env=entorno,
    )
    lineas = []
    for linea in proceso.stdout:
        lineas.append(linea)
        if verbose and not _RUIDO.match(linea):
            print(linea, end="")
    codigo = proceso.wait()
    if codigo != 0:
        utiles = [l for l in lineas if not _RUIDO.match(l)]
        raise RuntimeError(
            "El proceso de DisVoice terminó con error (código {}). Últimas líneas:\n{}".format(
                codigo, "".join(utiles[-25:]))
        )


def caracterizar_disvoice(
    entrada,
    dimensiones=("phonation",),
    estaticas=True,
    dinamicas=False,
    carpeta_salida="caracteristicas",
    sobrescribir=False,
    nan="preguntar",
    eliminar_filas_vacias=True,
    python_disvoice=None,
    verbose=True,
):
    """Extrae características de DisVoice en modo estático y/o dinámico, las limpia y las guarda.

    Parámetros
    - entrada: carpeta de audios (.wav, incluye subcarpetas) o lista de rutas
      (p. ej. seleccionar_datos(...)["seleccionados"]).
    - dimensiones: una o varias de DIMENSIONES ("phonation", "prosody",
      "articulation", "phonological", "glottal").
    - estaticas: un vector por audio (static=True de DisVoice).
    - dinamicas: un vector por frame o segmento (static=False de DisVoice).
      Ambos modos se pueden pedir a la vez: se extraen en el mismo proceso.
    - carpeta_salida: por defecto "caracteristicas", relativa al directorio de
      trabajo (en Spyder con %runfile --wdir, la carpeta del script). Guarda:
        <dim>.csv, <dim>_dyn.csv   características crudas (sin limpiar)
        disvoice_estaticas.pkl     {dimension: DataFrame} limpio
        disvoice_dinamicas.pkl     {dimension: DataFrame} limpio
        errores_disvoice.csv       registro de audios que fallaron
      None: no guarda nada.
    - sobrescribir: si es False, una dimensión cuyo CSV crudo ya existe se carga
      en vez de extraerse de nuevo (la limpieza y el pkl se rehacen siempre). Si
      ese CSV es de otra lista de audios, lanza ValueError.
    - nan, eliminar_filas_vacias: limpieza de NaN antes de guardar el pkl (ver
      gita.caracteristicas.limpiar_nan). nan="preguntar" muestra cuántas filas
      tienen NaN y pregunta en la consola qué hacer; también puede ser "media",
      "mediana", "moda", "cero" o "eliminar". Se aplica por separado a estáticas
      y dinámicas; un participante eliminado sale de todas las dimensiones.
    - python_disvoice: Python del entorno de DisVoice (ver docstring del módulo).

    phonological exige audios a 16 kHz (con otra frecuencia phonet devuelve NaN):
    si alguno no lo está, se lanza ValueError antes de extraer.

    Columnas: id (nombre sin extensión), ruta (distingue el mismo id en tareas
    distintas), [frame en dinámicas] y las características de DisVoice. En las
    crudas, los NaN de DisVoice se conservan y los ±inf pasan a NaN; un audio que
    falla queda con su fila en NaN (estáticas) o sin frames (dinámicas), y su
    error en "fallidos".

    Devuelve un dict con:
    - "estaticas": {dimension: DataFrame} limpio  (vacío si estaticas=False)
    - "dinamicas": {dimension: DataFrame} limpio  (vacío si dinamicas=False)
    - "crudas": {"estaticas": {...}, "dinamicas": {...}} sin limpiar
    - "limpieza": {tipo: informe de limpiar_nan (columnas e ids eliminados, estrategia)}
    - "fallidos": lista de dicts con dimension, tipo, ruta y error
    """
    if isinstance(dimensiones, str):
        dimensiones = [dimensiones]
    invalidas = [d for d in dimensiones if d not in DIMENSIONES]
    if invalidas:
        raise ValueError("Dimensiones no válidas: {}. Opciones: {}".format(invalidas, DIMENSIONES))
    tipos = [t for t, pedido in zip(TIPOS, (estaticas, dinamicas)) if pedido]
    if not tipos:
        raise ValueError("Pide al menos un tipo: estaticas=True y/o dinamicas=True")
    if nan != "preguntar" and nan not in ESTRATEGIAS:
        raise ValueError("nan debe ser 'preguntar' o una de {}".format(ESTRATEGIAS))

    rutas = [str(p.resolve()) for p in listar_archivos(entrada, extensiones=(".wav",))]
    if not rutas:
        raise ValueError("No hay archivos .wav en {}".format(entrada))

    # Errores de ejecuciones anteriores (necesarios para validar los CSV que se reutilizan)
    errores = pd.DataFrame(columns=COLUMNAS_ERRORES)
    if carpeta_salida is not None:
        carpeta_salida = Path(carpeta_salida).resolve()
        carpeta_salida.mkdir(parents=True, exist_ok=True)
        if verbose:
            print("Características en: {}".format(carpeta_salida))
        if (carpeta_salida / ARCHIVO_ERRORES).exists():
            errores = pd.read_csv(carpeta_salida / ARCHIVO_ERRORES)

    resultado = {"estaticas": {}, "dinamicas": {}}
    pendientes = {"estaticas": [], "dinamicas": []}

    # 1. Reutilizar los CSV que ya existan (si son de la misma lista de audios)
    for tipo in tipos:
        for dim in dimensiones:
            csv = carpeta_salida / _nombre_csv(dim, tipo) if carpeta_salida else None
            if csv is None or not csv.exists() or sobrescribir:
                pendientes[tipo].append(dim)
                continue
            df = pd.read_csv(csv)
            fallidas = errores.loc[(errores["dimension"] == dim) & (errores["tipo"] == tipo), "ruta"]
            if set(df["ruta"]) | set(fallidas) != set(rutas):
                raise ValueError(
                    "{} es de otra lista de audios. Usa sobrescribir=True u otra "
                    "carpeta_salida.".format(csv))
            resultado[tipo][dim] = df
            if verbose:
                print("[{} {}] ya existe {}: se carga sin volver a extraer".format(
                    dim, _ETIQUETA[tipo], csv.name))

    # 2. phonological necesita 16 kHz: verificar antes de lanzar el proceso
    if "phonological" in pendientes["estaticas"] + pendientes["dinamicas"]:
        no_16k = [r for r in rutas if sf.info(r).samplerate != 16000]
        if no_16k:
            raise ValueError(
                "phonological requiere audios a 16 kHz (con otra frecuencia phonet devuelve "
                "NaN). {} audios no lo están, p. ej. {}. Preprocésalos antes con "
                "gita.audio.preprocesar_carpeta.".format(len(no_16k), no_16k[0]))

    # 3. Extraer lo pendiente en un solo proceso
    if pendientes["estaticas"] or pendientes["dinamicas"]:
        python = _python_disvoice(python_disvoice)
        with tempfile.TemporaryDirectory() as tmp:
            _ejecutar_worker(python, rutas, pendientes, tmp, verbose)

            for tipo in TIPOS:
                for dim in pendientes[tipo]:
                    df = pd.read_csv(Path(tmp) / _nombre_csv(dim, tipo))
                    df.insert(0, "id", [Path(r).stem for r in df["ruta"]])
                    resultado[tipo][dim] = df
                    if carpeta_salida is not None:
                        df.to_csv(carpeta_salida / _nombre_csv(dim, tipo), index=False)

            # Reemplazar en el registro los errores de lo que se volvió a extraer
            nuevos = pd.read_csv(Path(tmp) / "errores.csv")
            rehechos = {(d, t) for t in TIPOS for d in pendientes[t]}
            viejos = errores[[(d, t) not in rehechos
                              for d, t in zip(errores["dimension"], errores["tipo"])]]
            errores = pd.concat([viejos, nuevos], ignore_index=True)
            if carpeta_salida is not None:
                errores.to_csv(carpeta_salida / ARCHIVO_ERRORES, index=False)

    pedidos = {(d, t) for t in tipos for d in dimensiones}
    fallidos = [
        fila for fila in errores[["dimension", "tipo", "ruta", "error"]].to_dict("records")
        if (fila["dimension"], fila["tipo"]) in pedidos
    ]
    crudas = {t: {d: resultado[t][d] for d in dimensiones} for t in tipos}

    # 4. Limpieza de NaN y pkl (por tipo)
    limpias, limpieza = {}, {}
    for tipo in tipos:
        if verbose:
            print("\n--- Limpieza de NaN: {} ---".format(_ETIQUETA[tipo]))
        informe = limpiar_nan(crudas[tipo], estrategia=nan,
                              eliminar_filas_vacias=eliminar_filas_vacias,
                              nombre=_ETIQUETA[tipo], verbose=verbose)
        limpias[tipo] = informe.pop("caracteristicas")
        limpieza[tipo] = informe
        if carpeta_salida is not None:
            pkl = guardar_pkl(limpias[tipo], carpeta_salida / "disvoice_{}.pkl".format(tipo))
            if verbose:
                print("Guardado: {}".format(pkl))

    return {
        "estaticas": limpias.get("estaticas", {}),
        "dinamicas": limpias.get("dinamicas", {}),
        "crudas": crudas,
        "limpieza": limpieza,
        "fallidos": fallidos,
    }
