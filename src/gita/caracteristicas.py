"""
Manejo de matrices de características: limpieza de NaN, IDs, selección, clases y pkl.

Funciona con cualquier tipo de característica (audio, IMU, EEG...). Todas las
funciones aceptan:
- un DataFrame,
- un dict {dimension: DataFrame} (p. ej. {"phonation": df, "prosody": df}),
- o la ruta a un .pkl que contenga cualquiera de los dos,
y devuelven la misma estructura que reciben.

Columnas de metadatos (no son características): "id" (participante),
"ruta" (archivo de origen) y "frame" (solo en características dinámicas, una
fila por frame). Cada DataFrame debe tener la columna "id".
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import leer_ids

METADATOS = ("id", "ruta", "frame")
ESTRATEGIAS = ("media", "mediana", "moda", "cero", "eliminar")


# --- utilidades internas ----------------------------------------------------

def _a_dict(caracteristicas):
    """Devuelve (dict {dim: DataFrame}, era_dataframe)."""
    if isinstance(caracteristicas, (str, Path)):
        caracteristicas = cargar_pkl(caracteristicas)
    if isinstance(caracteristicas, pd.DataFrame):
        return {"_": caracteristicas}, True
    if isinstance(caracteristicas, dict):
        return dict(caracteristicas), False
    raise TypeError("Se esperaba un DataFrame, un dict {dimension: DataFrame} o la ruta a un .pkl")


def _restaurar(dims, era_dataframe):
    return dims["_"] if era_dataframe else dims


def _columnas(df):
    return [c for c in df.columns if c not in METADATOS]


def _eliminar_ids(dims, ids):
    for d in dims:
        dims[d] = dims[d][~dims[d]["id"].isin(ids)].reset_index(drop=True)


def _sincronizar(dims):
    """Deja solo los participantes presentes en TODAS las dimensiones. Devuelve los quitados."""
    conjuntos = [set(df["id"]) for df in dims.values()]
    todos, comunes = set().union(*conjuntos), set.intersection(*conjuntos)
    quitar = todos - comunes
    if quitar:
        _eliminar_ids(dims, quitar)
    return quitar


def _preguntar_estrategia(etiqueta):
    opciones = " / ".join(ESTRATEGIAS)
    while True:
        respuesta = input("{}¿Qué hacer con los NaN? [{}]: ".format(etiqueta, opciones)).strip().lower()
        if respuesta in ESTRATEGIAS:
            return respuesta
        print("Opción no válida: {!r}".format(respuesta))


# --- pkl --------------------------------------------------------------------

def guardar_pkl(caracteristicas, ruta):
    """Guarda un DataFrame o un dict {dimension: DataFrame} en un .pkl. Crea las carpetas."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as f:
        pickle.dump(caracteristicas, f, protocol=4)   # protocolo 4: legible desde Python 3.4+
    return ruta


def cargar_pkl(ruta):
    """Carga un .pkl guardado con guardar_pkl."""
    with open(ruta, "rb") as f:
        return pickle.load(f)


# --- limpieza de NaN --------------------------------------------------------

def limpiar_nan(caracteristicas, estrategia="preguntar", eliminar_filas_vacias=True, nombre="",
                verbose=True):
    """Limpia los NaN de una matriz (o dict de matrices) de características.

    Pasos:
    1. Elimina las columnas (características) que son NaN en todas las filas.
    2. Si eliminar_filas_vacias=True, elimina las filas que son NaN en todas sus
       características (p. ej. audios que fallaron por completo): imputarlas
       sería inventar un participante.
    3. Informa cuántas filas siguen teniendo NaN y aplica la estrategia:
       - "media", "mediana", "moda" o "cero": imputa cada columna con ese valor
         (calculado en esa misma dimensión);
       - "eliminar": estáticas -> elimina a los participantes con algún NaN;
         dinámicas -> elimina los frames con algún NaN;
       - "preguntar" (por defecto): muestra el informe y pregunta en la consola.
    En todos los casos, un participante eliminado de una dimensión se elimina
    de todas (todas las dimensiones quedan con los mismos participantes).

    Nota: imputar con estadísticos de todos los datos antes de dividir en
    entrenamiento/prueba filtra información de prueba al entrenamiento. Para los
    experimentos finales conviene imputar dentro de cada partición.

    Devuelve un dict con:
    - "caracteristicas": la misma estructura de entrada, limpia
    - "columnas_eliminadas": {dimension: [columnas]}
    - "ids_eliminados": participantes eliminados (de todas las dimensiones)
    - "estrategia": la estrategia aplicada ("ninguna" si no quedaban NaN)
    """
    if estrategia != "preguntar" and estrategia not in ESTRATEGIAS:
        raise ValueError("estrategia debe ser 'preguntar' o una de {}".format(ESTRATEGIAS))

    dims, era_dataframe = _a_dict(caracteristicas)
    dims = {d: df.copy() for d, df in dims.items()}
    dinamico = any("frame" in df.columns for df in dims.values())
    etiqueta = "[{}] ".format(nombre) if nombre else ""
    unidad = "frames" if dinamico else "filas"
    ids_iniciales = set().union(*[set(df["id"]) for df in dims.values()])

    def mostrar(texto):
        if verbose:
            print(etiqueta + texto)

    # 1. Columnas todo-NaN
    columnas_eliminadas = {}
    for d, df in dims.items():
        vacias = [c for c in _columnas(df) if df[c].isna().all()]
        dims[d] = df.drop(columns=vacias)
        columnas_eliminadas[d] = vacias
        if vacias:
            mostrar("{}: {} columnas todo-NaN eliminadas: {}".format(d, len(vacias), vacias))

    # 2. Filas todo-NaN (audios fallidos)
    if eliminar_filas_vacias:
        for d, df in dims.items():
            vacias = df[_columnas(df)].isna().all(axis=1)
            if vacias.any():
                mostrar("{}: {} {} todo-NaN eliminadas".format(d, int(vacias.sum()), unidad))
                dims[d] = df[~vacias].reset_index(drop=True)
    _sincronizar(dims)

    # 3. Informe de los NaN restantes
    afectados = set()
    hay_nan = False
    n_participantes = len(set(next(iter(dims.values()))["id"])) if dims else 0
    for d, df in dims.items():
        con_nan = df[_columnas(df)].isna().any(axis=1)
        n_valores = int(df[_columnas(df)].isna().to_numpy().sum())
        afectados |= set(df.loc[con_nan, "id"])
        hay_nan = hay_nan or bool(con_nan.any())
        mostrar("{}: {} de {} {} con NaN ({} valores NaN)".format(
            d, int(con_nan.sum()), len(df), unidad, n_valores))

    if not hay_nan:
        estrategia_aplicada = "ninguna"
        mostrar("No quedan NaN.")
    else:
        if dinamico:
            mostrar("{} de {} participantes tienen frames con NaN (con 'eliminar' se quitan esos "
                    "frames, no el participante)".format(len(afectados), n_participantes))
        else:
            mostrar("{} de {} participantes tienen algún NaN (con 'eliminar' quedarían {})".format(
                len(afectados), n_participantes, n_participantes - len(afectados)))

        if estrategia == "preguntar":
            estrategia = _preguntar_estrategia(etiqueta)
        estrategia_aplicada = estrategia

        # 4. Aplicar la estrategia
        if estrategia == "eliminar":
            if dinamico:
                for d, df in dims.items():
                    dims[d] = df[~df[_columnas(df)].isna().any(axis=1)].reset_index(drop=True)
            else:
                _eliminar_ids(dims, afectados)
        else:
            for d, df in dims.items():
                for c in _columnas(df):
                    if not df[c].isna().any():
                        continue
                    if estrategia == "media":
                        valor = df[c].mean()
                    elif estrategia == "mediana":
                        valor = df[c].median()
                    elif estrategia == "moda":
                        valor = df[c].mode().iloc[0]
                    else:
                        valor = 0.0
                    df[c] = df[c].fillna(valor)
        _sincronizar(dims)
        mostrar("Estrategia aplicada: {}".format(estrategia))

    ids_finales = set(next(iter(dims.values()))["id"]) if dims else set()
    ids_eliminados = sorted(ids_iniciales - ids_finales)
    if ids_eliminados:
        mostrar("{} participantes eliminados de todas las dimensiones: {}{}".format(
            len(ids_eliminados), ids_eliminados[:10], " ..." if len(ids_eliminados) > 10 else ""))
    for d, df in dims.items():
        mostrar("{}: {} {} x {} características".format(d, len(df), unidad, len(_columnas(df))))

    return {
        "caracteristicas": _restaurar(dims, era_dataframe),
        "columnas_eliminadas": _restaurar(columnas_eliminadas, era_dataframe),
        "ids_eliminados": ids_eliminados,
        "estrategia": estrategia_aplicada,
    }


# --- IDs y selección --------------------------------------------------------

def obtener_ids(caracteristicas, ruta_txt=None):
    """IDs de los participantes presentes en TODAS las dimensiones.

    Conserva el orden en que aparecen en la primera dimensión. Si se da
    ruta_txt, los guarda ahí, uno por línea (se pueden leer con leer_ids).
    """
    dims, _ = _a_dict(caracteristicas)
    comunes = set.intersection(*[set(df["id"]) for df in dims.values()])
    primera = next(iter(dims.values()))
    ids = [i for i in dict.fromkeys(primera["id"]) if i in comunes]
    if ruta_txt is not None:
        ruta_txt = Path(ruta_txt)
        ruta_txt.parent.mkdir(parents=True, exist_ok=True)
        ruta_txt.write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8")
    return ids


def seleccionar_caracteristicas(caracteristicas, ids, columna=None, verbose=True):
    """Filtra una matriz (o dict de matrices) de características a una lista de IDs.

    Equivalente a gita.datos.seleccionar_datos, pero sobre características en
    vez de archivos.
    - ids: ruta a un archivo de IDs (.txt uno por línea, o .csv/.tsv con
      columna) o una lista de IDs.

    Si ningún ID coincide, lanza ValueError.

    Devuelve un dict con "caracteristicas" (misma estructura de entrada,
    filtrada) e "ids_no_encontrados" (IDs de la lista que no están en ninguna
    dimensión).
    """
    if isinstance(ids, (str, Path)):
        ids = leer_ids(ids, columna)
    ids = [str(i).strip() for i in ids]

    dims, era_dataframe = _a_dict(caracteristicas)
    filtrado = {}
    for d, df in dims.items():
        filtrado[d] = df[df["id"].astype(str).isin(ids)].reset_index(drop=True)

    encontrados = set().union(*[set(df["id"].astype(str)) for df in filtrado.values()])
    if not encontrados:
        raise ValueError("Ningún ID coincide (p. ej. {}). Revisa la columna o el formato de los "
                         "IDs.".format(ids[:3]))
    no_encontrados = [i for i in dict.fromkeys(ids) if i not in encontrados]

    if verbose:
        print("{} participantes seleccionados de {} IDs".format(len(encontrados), len(set(ids))))
        if not era_dataframe:
            for d, df in filtrado.items():
                print("  {}: {} filas".format(d, len(df)))
        if no_encontrados:
            print("IDs no encontrados ({}): {}{}".format(
                len(no_encontrados), no_encontrados[:10], " ..." if len(no_encontrados) > 10 else ""))

    return {"caracteristicas": _restaurar(filtrado, era_dataframe), "ids_no_encontrados": no_encontrados}


def obtener_clases(columna, patrones, regex=False, ignorar_mayusculas=False, sin_clase="error"):
    """Asigna una clase a cada fila según el texto que contiene la columna.

    Equivale a aplicar ``columna.str.contains(patron)`` para cada clase y unir
    los resultados en un solo vector de etiquetas.

    Parameters
    ----------
    columna : pandas.Series o array-like
        Valores de los que se extrae la clase (p. ej. ``df["id"]``).
    patrones : dict {etiqueta: patron} o lista de str
        Texto que identifica cada clase. Con una lista, la etiqueta es el
        mismo patrón: ``["GG", "GU"]`` equivale a ``{"GG": "GG", "GU": "GU"}``.
    regex : bool
        Si True, los patrones son expresiones regulares (p. ej. ``"^GG"``,
        ``"_CTRL$"``). Por defecto se buscan como texto literal.
    ignorar_mayusculas : bool
        Si True, ``"gg"`` coincide con ``"GG"``.
    sin_clase : {"error", "nan"}
        Qué hacer con las filas que no coinciden con ningún patrón:
        ``"error"`` lanza un ValueError; ``"nan"`` las deja en NaN (para
        filtrarlas después con ``y.notna()``).

    Returns
    -------
    pandas.Series
        Etiquetas, con el mismo índice que ``columna`` y nombre ``"clase"``.

    Raises
    ------
    ValueError
        Si una fila coincide con más de una clase (patrones ambiguos), o si
        alguna fila no tiene clase y ``sin_clase="error"``.
    """
    if sin_clase not in ("error", "nan"):
        raise ValueError("sin_clase debe ser 'error' o 'nan'.")
    if not isinstance(patrones, dict):
        patrones = {p: p for p in patrones}
    if not patrones:
        raise ValueError("Se necesita al menos un patrón.")

    valores = columna if isinstance(columna, pd.Series) else pd.Series(columna)
    valores = valores.astype(str)

    coincide = pd.DataFrame({
        etiqueta: valores.str.contains(patron, regex=regex, case=not ignorar_mayusculas)
        for etiqueta, patron in patrones.items()
    })
    n_coincidencias = coincide.sum(axis=1)

    ambiguos = valores[n_coincidencias > 1]
    if len(ambiguos):
        ejemplos = ", ".join(
            f"{v!r} -> {list(coincide.columns[coincide.loc[i]])}"
            for i, v in ambiguos.head(5).items()
        )
        raise ValueError(f"{len(ambiguos)} valores coinciden con más de una clase: {ejemplos}")

    sin = valores[n_coincidencias == 0]
    if len(sin) and sin_clase == "error":
        raise ValueError(
            f"{len(sin)} valores no coinciden con ninguna clase: {list(sin.head(10))}"
            " (usa sin_clase='nan' para dejarlos en NaN)"
        )

    clases = pd.Series(np.nan, index=valores.index, dtype=object, name="clase")
    for etiqueta in coincide.columns:
        clases[coincide[etiqueta]] = etiqueta
    return clases
