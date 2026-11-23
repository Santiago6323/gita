import pytest

from gita.datos import (
    crear_enlaces,
    interseccion_ids,
    leer_ids,
    listar_archivos,
    seleccionar_datos,
    seleccionar_por_ids,
)


# --- leer_ids ---------------------------------------------------------------

def test_leer_ids_txt(tmp_path):
    ruta = tmp_path / "ids.txt"
    ruta.write_text("ID001\n  ID002 \n\n# comentario\nID001\nID003\n")
    assert leer_ids(ruta) == ["ID001", "ID002", "ID003"]


@pytest.mark.parametrize("sep", [",", ";", "\t"])
def test_leer_ids_csv_detecta_separador(tmp_path, sep):
    ruta = tmp_path / "ids.csv"
    filas = ["id{0}grupo".format(sep), "ID001{0}GG".format(sep), "ID002{0}GU".format(sep)]
    ruta.write_text("\n".join(filas) + "\n")
    assert leer_ids(ruta, columna="id") == ["ID001", "ID002"]


def test_leer_ids_csv_con_bom_de_excel(tmp_path):
    ruta = tmp_path / "ids.csv"
    ruta.write_bytes("id;grupo\nID001;GG\n".encode("utf-8-sig"))
    assert leer_ids(ruta, columna="id") == ["ID001"]


def test_leer_ids_columna_inexistente(tmp_path):
    ruta = tmp_path / "ids.csv"
    ruta.write_text("id,grupo\nID001,GG\n")
    with pytest.raises(KeyError):
        leer_ids(ruta, columna="sujeto")


# --- seleccionar_por_ids ----------------------------------------------------

@pytest.fixture
def base(tmp_path):
    carpeta = tmp_path / "audios"
    for tarea, ids in {"tarea1": ["ID001", "ID002", "ID003"], "tarea2": ["ID001", "ID003"]}.items():
        (carpeta / tarea).mkdir(parents=True)
        for i in ids:
            (carpeta / tarea / (i + ".wav")).write_bytes(b"")
    (carpeta / "tarea1" / "ID001.txt").write_text("no es audio")
    return carpeta


def test_selecciona_en_todas_las_tareas(base):
    r = seleccionar_por_ids(base, ["ID001", "ID003"], verbose=False)
    nombres = [str(p.relative_to(base)) for p in r["seleccionados"]]
    assert nombres == [
        "tarea1/ID001.wav", "tarea1/ID003.wav", "tarea2/ID001.wav", "tarea2/ID003.wav",
    ]
    assert r["ids_no_encontrados"] == []


def test_reporta_ids_no_encontrados(base):
    r = seleccionar_por_ids(base, ["ID002", "ID999", "1"], verbose=False)
    assert len(r["seleccionados"]) == 1
    assert r["ids_no_encontrados"] == ["ID999", "1"]


def test_no_copia_nada(base):
    antes = sorted(base.rglob("*"))
    seleccionar_por_ids(base, ["ID001"], verbose=False)
    assert sorted(base.rglob("*")) == antes


def test_carpeta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        seleccionar_por_ids(tmp_path / "no_existe", ["ID001"])


# --- crear_enlaces ----------------------------------------------------------

def test_crear_enlaces_estructura_y_contenido(base, tmp_path):
    (base / "tarea1" / "ID001.wav").write_bytes(b"datos de audio")
    r = seleccionar_por_ids(base, ["ID001", "ID003"], verbose=False)
    destino = tmp_path / "seleccionados"

    enlaces = crear_enlaces(r["seleccionados"], base, destino, verbose=False)

    assert len(enlaces) == 4
    assert all(e.is_symlink() for e in enlaces)
    assert (destino / "tarea1" / "ID001.wav").read_bytes() == b"datos de audio"
    assert (destino / "tarea2" / "ID003.wav").resolve() == (base / "tarea2" / "ID003.wav").resolve()


def test_crear_enlaces_reemplaza_seleccion_anterior(base, tmp_path):
    destino = tmp_path / "seleccionados"
    crear_enlaces(seleccionar_por_ids(base, ["ID001", "ID002", "ID003"], verbose=False)["seleccionados"],
                  base, destino, verbose=False)
    crear_enlaces(seleccionar_por_ids(base, ["ID002"], verbose=False)["seleccionados"],
                  base, destino, verbose=False)

    restantes = sorted(str(p.relative_to(destino)) for p in destino.rglob("*.wav"))
    assert restantes == ["tarea1/ID002.wav"]          # los enlaces viejos ya no están
    assert not (destino / "tarea2").exists()          # ni las carpetas que quedaron vacías
    assert (base / "tarea1" / "ID001.wav").exists()   # los originales siguen ahí


def test_crear_enlaces_no_toca_archivos_reales(base, tmp_path):
    destino = tmp_path / "seleccionados"
    destino.mkdir()
    (destino / "importante.wav").write_bytes(b"real")
    with pytest.raises(FileExistsError):
        crear_enlaces([base / "tarea1" / "ID001.wav"], base, destino, verbose=False)
    assert (destino / "importante.wav").read_bytes() == b"real"


def test_crear_enlaces_destino_dentro_de_base(base):
    with pytest.raises(ValueError):
        crear_enlaces([base / "tarea1" / "ID001.wav"], base, base / "seleccionados", verbose=False)


def test_crear_enlaces_ruta_fuera_de_base(base, tmp_path):
    otro = tmp_path / "otro.wav"
    otro.write_bytes(b"")
    with pytest.raises(ValueError):
        crear_enlaces([otro], base, tmp_path / "seleccionados", verbose=False)


def test_preprocesar_carpeta_acepta_carpeta_de_enlaces(base, tmp_path):
    import numpy as np
    import soundfile as sf
    from gita.audio import preprocesar_carpeta

    for p in base.rglob("*.wav"):
        sf.write(str(p), 0.3 * np.sin(np.linspace(0, 100, 44100)), 44100)
    destino = tmp_path / "seleccionados"
    crear_enlaces(seleccionar_por_ids(base, ["ID001"], verbose=False)["seleccionados"],
                  base, destino, verbose=False)

    r = preprocesar_carpeta(destino, tmp_path / "salida", verbose=False)
    assert len(r["procesados"]) == 2


# --- seleccionar_datos ------------------------------------------------------

def test_seleccionar_datos_desde_csv(base, tmp_path):
    ids = tmp_path / "ids.csv"
    ids.write_text("id;grupo\nID001;GG\nID003;GU\nID999;GG\n")
    destino = tmp_path / "seleccionados"

    r = seleccionar_datos(base, ids, destino, columna="id", verbose=False)

    assert len(r["seleccionados"]) == 4 and len(r["enlaces"]) == 4
    assert r["ids_no_encontrados"] == ["ID999"]
    assert sorted(str(p.relative_to(destino)) for p in destino.rglob("*.wav")) == [
        "tarea1/ID001.wav", "tarea1/ID003.wav", "tarea2/ID001.wav", "tarea2/ID003.wav",
    ]


def test_seleccionar_datos_sin_destino_no_crea_nada(base, tmp_path):
    antes = sorted(tmp_path.rglob("*"))
    r = seleccionar_datos(base, ["ID001", "ID003"], verbose=False)
    assert len(r["seleccionados"]) == 4 and r["enlaces"] is None
    assert sorted(tmp_path.rglob("*")) == antes


def test_seleccionar_datos_con_lista_y_destino(base, tmp_path):
    r = seleccionar_datos(base, ["ID002"], tmp_path / "sel", verbose=False)
    assert len(r["enlaces"]) == 1


def test_seleccionar_datos_sin_coincidencias_no_toca_destino(base, tmp_path):
    destino = tmp_path / "seleccionados"
    seleccionar_datos(base, ["ID001"], destino, verbose=False)
    antes = sorted(destino.rglob("*"))

    with pytest.raises(ValueError):
        seleccionar_datos(base, ["001", "002"], destino, verbose=False)
    assert sorted(destino.rglob("*")) == antes   # la selección anterior sigue intacta


# --- listar_archivos --------------------------------------------------------

def test_listar_archivos_carpeta(base):
    archivos = listar_archivos(base)
    assert len(archivos) == 5 and all(p.suffix == ".wav" for p in archivos)
    assert archivos == sorted(archivos)


def test_listar_archivos_no_recursivo_y_extensiones(base):
    assert listar_archivos(base, recursivo=False) == []
    assert len(listar_archivos(base, extensiones=(".txt",))) == 1


def test_listar_archivos_archivo_unico(base):
    ruta = base / "tarea1" / "ID001.wav"
    assert listar_archivos(str(ruta)) == [ruta]


def test_listar_archivos_lista(base):
    lista = [str(base / "tarea2" / "ID003.wav"), base / "tarea1" / "ID001.wav"]
    assert listar_archivos(lista) == [base / "tarea2" / "ID003.wav", base / "tarea1" / "ID001.wav"]


def test_listar_archivos_lista_con_ruta_inexistente(base):
    with pytest.raises(FileNotFoundError):
        listar_archivos([base / "tarea1" / "ID001.wav", base / "no_existe.wav"])


def test_listar_archivos_carpeta_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        listar_archivos(tmp_path / "no_existe")


def test_seleccion_se_pasa_directo_a_listar_archivos(base):
    r = seleccionar_datos(base, ["ID001"], verbose=False)
    assert listar_archivos(r["seleccionados"]) == [
        base / "tarea1" / "ID001.wav", base / "tarea2" / "ID001.wav",
    ]


# --- interseccion_ids -------------------------------------------------------

def test_interseccion_ids_ejemplo():
    ids = interseccion_ids(["CTR01", "CTR02", "CTR03"],
                           ["CTR01", "CTR04", "CTR03"],
                           ["CTR01", "CTR03", "CTR05"], verbose=False)
    assert ids == ["CTR01", "CTR03"]


def test_interseccion_ids_mezcla_listas_txt_y_csv(tmp_path):
    txt = tmp_path / "a.txt"
    txt.write_text("CTR03\nCTR01\nCTR02\n")
    csv = tmp_path / "b.csv"
    csv.write_text("sujeto;grupo\nCTR01;GG\nCTR03;GU\nCTR09;GG\n")
    salida = tmp_path / "comunes.txt"

    ids = interseccion_ids(txt, (csv, "sujeto"), ["CTR01", "CTR03", " CTR03 "],
                           ruta_txt=salida, verbose=False)

    assert ids == ["CTR03", "CTR01"]                  # orden de la primera lista
    assert leer_ids(salida) == ["CTR03", "CTR01"]


def test_interseccion_ids_vacia():
    assert interseccion_ids(["CTR01"], ["CTR02"], verbose=False) == []


def test_interseccion_ids_requiere_dos_listas():
    with pytest.raises(ValueError):
        interseccion_ids(["CTR01"])
