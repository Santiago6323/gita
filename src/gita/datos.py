"""
Selección de archivos de una base de datos a partir de una lista de IDs.

El ID de un archivo es su nombre sin extensión: tarea1/ID001.wav -> "ID001".
Nada aquí lee el contenido de los archivos: solo trabaja con rutas, así que
sirve igual para audio, IMU, EEG, etc.
"""

import csv
import os
from collections import Counter
from pathlib import Path


def listar_archivos(entrada, extensiones=(".wav",), recursivo=True):
    """Normaliza la entrada de las funciones que procesan varios archivos.

    entrada puede ser:
    - una carpeta: devuelve sus archivos con esas extensiones (recursivo=True
      incluye subcarpetas), ordenados;
    - un archivo: devuelve [ese archivo];
    - una lista (o cualquier iterable) de rutas: la devuelve como lista de Path,
      sin filtrar por extensión (se asume que la selección ya es la deseada).

    Lanza FileNotFoundError si la carpeta, el archivo o alguna ruta de la
    lista no existe.
    """
    if isinstance(entrada, (str, Path)):
        entrada = Path(entrada)
        if entrada.is_dir():
            extensiones = {e.lower() for e in extensiones}
            patron = entrada.rglob("*") if recursivo else entrada.glob("*")
            return sorted(p for p in patron if p.is_file() and p.suffix.lower() in extensiones)
        if entrada.is_file():
            return [entrada]
        raise FileNotFoundError("No existe: {}".format(entrada))

    archivos = [Path(p) for p in entrada]
    faltantes = [p for p in archivos if not p.is_file()]
    if faltantes:
        raise FileNotFoundError(
            "{} rutas de la lista no existen, p. ej. {}".format(len(faltantes), faltantes[0])
        )
    return archivos


def leer_ids(ruta, columna=None):
    """Lee una lista de IDs desde un archivo de texto.

    - columna=None: un ID por línea (.txt). Se ignoran líneas vacías y las que
      empiezan con '#'.
    - columna="nombre": archivo con encabezado (.csv, .tsv, ...); se toma esa
      columna. El separador (',', ';' o tabulación) se detecta solo.

    Quita espacios y duplicados (conserva el orden). Devuelve una lista de str.
    """
    # utf-8-sig: ignora el BOM que agrega Excel al guardar CSV
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        texto = f.read()

    if columna is None:
        valores = [l.strip() for l in texto.splitlines()]
        valores = [v for v in valores if v and not v.startswith("#")]
    else:
        try:
            dialecto = csv.Sniffer().sniff(texto[:4096], delimiters=",;\t")
            separador = dialecto.delimiter
        except csv.Error:  # una sola columna: no hay separador que detectar
            separador = ","
        lector = csv.DictReader(texto.splitlines(), delimiter=separador)
        columnas = [c.strip() for c in (lector.fieldnames or [])]
        if columna not in columnas:
            raise KeyError(
                "No existe la columna '{}'. Columnas disponibles: {}".format(columna, columnas)
            )
        lector.fieldnames = columnas
        valores = [(fila[columna] or "").strip() for fila in lector]
        valores = [v for v in valores if v]

    return list(dict.fromkeys(valores))  # quita duplicados conservando el orden


def interseccion_ids(*listas, ruta_txt=None, verbose=True):
    """IDs presentes en TODAS las listas (inner join).

        interseccion_ids(["CTR01", "CTR02", "CTR03"],
                         ["CTR01", "CTR04", "CTR03"],
                         ["CTR01", "CTR03", "CTR05"])   ->  ["CTR01", "CTR03"]

    Cada lista puede ser:
    - una lista (o cualquier iterable) de IDs, p. ej. la salida de obtener_ids;
    - la ruta a un .txt con un ID por línea;
    - una tupla (ruta, columna) para un .csv/.tsv con encabezado.

    Conserva el orden de la primera lista. Si se da ruta_txt, guarda el
    resultado ahí, uno por línea (se puede leer con leer_ids).
    """
    if len(listas) < 2:
        raise ValueError("Se necesitan al menos dos listas de IDs")

    leidas = []
    for lista in listas:
        if isinstance(lista, (str, Path)):
            lista = leer_ids(lista)
        elif isinstance(lista, tuple) and len(lista) == 2 and isinstance(lista[0], (str, Path)):
            lista = leer_ids(lista[0], columna=lista[1])
        leidas.append(list(dict.fromkeys(str(i).strip() for i in lista)))

    comunes = set.intersection(*[set(l) for l in leidas])
    ids = [i for i in leidas[0] if i in comunes]

    if verbose:
        for k, lista in enumerate(leidas, 1):
            print("Lista {}: {} IDs ({} no están en todas)".format(k, len(lista), len(lista) - len(ids)))
        print("En todas las listas: {} IDs".format(len(ids)))

    if ruta_txt is not None:
        ruta_txt = Path(ruta_txt)
        ruta_txt.parent.mkdir(parents=True, exist_ok=True)
        ruta_txt.write_text("\n".join(ids) + "\n", encoding="utf-8")
    return ids


def seleccionar_por_ids(carpeta, ids, extensiones=(".wav",), recursivo=True, verbose=True):
    """Busca en carpeta los archivos cuyo nombre (sin extensión) está en ids.

    No copia nada: devuelve rutas a los archivos originales.

    Devuelve un dict con:
    - "seleccionados": lista de Path de los archivos encontrados (ordenada)
    - "ids_no_encontrados": IDs de la lista sin ningún archivo en la carpeta
      (casi siempre indica un error de nombre, p. ej. "001" vs "1")
    """
    carpeta = Path(carpeta)
    if not carpeta.is_dir():
        raise FileNotFoundError("No existe la carpeta: {}".format(carpeta))

    ids = [str(i).strip() for i in ids]
    buscados = set(ids)
    archivos = listar_archivos(carpeta, extensiones, recursivo)
    seleccionados = [p for p in archivos if p.stem in buscados]
    encontrados = {p.stem for p in seleccionados}
    no_encontrados = [i for i in dict.fromkeys(ids) if i not in encontrados]

    if verbose:
        print("{} archivos seleccionados de {} IDs".format(len(seleccionados), len(buscados)))
        por_carpeta = Counter(str(p.parent.relative_to(carpeta)) for p in seleccionados)
        for sub, n in sorted(por_carpeta.items()):
            print("  {}: {} de {}".format(sub, n, len(buscados)))
        if no_encontrados:
            muestra = ", ".join(no_encontrados[:10])
            extra = " ..." if len(no_encontrados) > 10 else ""
            print("IDs sin ningún archivo ({}): {}{}".format(len(no_encontrados), muestra, extra))

    return {"seleccionados": seleccionados, "ids_no_encontrados": no_encontrados}


def _esta_dentro(ruta, carpeta):
    """True si ruta está dentro de carpeta (o es la misma). Compatible con Python 3.8."""
    try:
        Path(ruta).resolve().relative_to(Path(carpeta).resolve())
        return True
    except ValueError:
        return False


def crear_enlaces(rutas, carpeta_base, carpeta_destino, verbose=True):
    """Crea en carpeta_destino enlaces simbólicos a los archivos de rutas.

    Mantiene la estructura relativa a carpeta_base:
        carpeta_base/tarea1/ID001.wav -> carpeta_destino/tarea1/ID001.wav (enlace)

    Los enlaces no duplican los datos: abrir un enlace es leer el original.
    Pensado para pasar a la caracterización una carpeta con solo los archivos
    seleccionados (p. ej. con seleccionar_por_ids).

    carpeta_destino queda exactamente con la selección actual: antes de crear
    los enlaces se borran los enlaces de una ejecución anterior (nunca los
    originales). Si carpeta_destino contiene archivos reales (no enlaces), no se
    toca nada y se lanza FileExistsError.

    Nota: en Windows crear enlaces simbólicos requiere permisos especiales.

    Devuelve la lista de rutas de los enlaces creados.
    """
    carpeta_base = Path(carpeta_base).resolve()
    carpeta_destino = Path(carpeta_destino)
    if _esta_dentro(carpeta_destino, carpeta_base):
        raise ValueError("La carpeta destino no puede estar dentro de la carpeta base")

    # 1. Calcular destinos (valida que todas las rutas estén dentro de carpeta_base)
    pares = []
    for ruta in rutas:
        original = Path(ruta).resolve()
        try:
            relativa = original.relative_to(carpeta_base)
        except ValueError:
            raise ValueError("{} no está dentro de {}".format(original, carpeta_base))
        pares.append((original, carpeta_destino / relativa))

    # 2. Limpiar la ejecución anterior (solo enlaces; si hay archivos reales, no tocar nada)
    if carpeta_destino.exists():
        contenido = list(carpeta_destino.rglob("*"))
        reales = [p for p in contenido if p.is_file() and not p.is_symlink()]
        if reales:
            raise FileExistsError(
                "{} contiene archivos reales (no enlaces), p. ej. {}. "
                "Usa una carpeta destino vacía o solo de enlaces.".format(carpeta_destino, reales[0])
            )
        for p in contenido:
            if p.is_symlink():
                p.unlink()
        for d in sorted(carpeta_destino.rglob("*"), reverse=True):  # subcarpetas vacías
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()

    # 3. Crear los enlaces (apuntan a rutas absolutas)
    creados = []
    for original, enlace in pares:
        enlace.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(original), str(enlace))
        creados.append(enlace)

    if verbose:
        print("{} enlaces creados en {}".format(len(creados), carpeta_destino))
    return creados


def seleccionar_datos(
    carpeta, ids, carpeta_destino=None, columna=None, extensiones=(".wav",), recursivo=True,
    verbose=True,
):
    """Selecciona los archivos de carpeta cuyos nombres están en una lista de IDs.

    Encadena leer_ids -> seleccionar_por_ids (-> crear_enlaces, opcional).

    - ids: ruta a un archivo de IDs (str o Path; se lee con leer_ids y columna)
      o una lista de IDs.
    - carpeta_destino: None (por defecto) no crea nada en disco; la lista de
      rutas en "seleccionados" se pasa directamente a las funciones que
      procesan archivos. Si se da una ruta, crea ahí una carpeta de enlaces
      simbólicos (para herramientas externas que solo acepten carpetas).

    Si ningún archivo coincide con los IDs, lanza ValueError sin tocar nada
    (suele indicar una columna o unos nombres equivocados).

    Devuelve un dict con "seleccionados" (rutas originales),
    "ids_no_encontrados" y "enlaces" (None si no se creó carpeta de enlaces).
    """
    if isinstance(ids, (str, Path)):
        ids = leer_ids(ids, columna)

    r = seleccionar_por_ids(carpeta, ids, extensiones, recursivo, verbose)
    if not r["seleccionados"]:
        raise ValueError(
            "Ningún archivo de {} coincide con los IDs (p. ej. {}). "
            "Revisa la columna o el formato de los IDs.".format(carpeta, list(ids)[:3])
        )

    enlaces = None
    if carpeta_destino is not None:
        enlaces = crear_enlaces(r["seleccionados"], carpeta, carpeta_destino, verbose)
    return {
        "seleccionados": r["seleccionados"],
        "ids_no_encontrados": r["ids_no_encontrados"],
        "enlaces": enlaces,
    }
