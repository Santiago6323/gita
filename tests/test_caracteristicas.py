import numpy as np
import pandas as pd
import pytest

from gita.caracteristicas import (
    cargar_pkl,
    guardar_pkl,
    limpiar_nan,
    obtener_clases,
    obtener_ids,
    seleccionar_caracteristicas,
)
from gita.datos import leer_ids

NAN = np.nan


@pytest.fixture
def estaticas():
    """Dos dimensiones. ID3 falla en phonation (fila todo-NaN); ID2 tiene un NaN suelto en prosody."""
    phonation = pd.DataFrame({
        "id": ["ID1", "ID2", "ID3", "ID4"],
        "ruta": ["a/ID1.wav", "a/ID2.wav", "a/ID3.wav", "a/ID4.wav"],
        "f0": [1.0, 2.0, NAN, 4.0],
        "jitter": [1.0, 1.0, NAN, 3.0],
        "vacia": [NAN, NAN, NAN, NAN],            # columna todo-NaN
    })
    prosody = pd.DataFrame({
        "id": ["ID1", "ID2", "ID3", "ID4"],
        "ruta": ["a/ID1.wav", "a/ID2.wav", "a/ID3.wav", "a/ID4.wav"],
        "energia": [10.0, NAN, 30.0, 50.0],
        "duracion": [1.0, 2.0, 3.0, 4.0],       # así el NaN de ID2 es parcial, no fila vacía
    })
    return {"phonation": phonation, "prosody": prosody}


@pytest.fixture
def dinamicas():
    """ID1: 3 frames (uno con NaN). ID2: 2 frames, ambos con NaN en phonation."""
    phonation = pd.DataFrame({
        "id": ["ID1", "ID1", "ID1", "ID2", "ID2"],
        "ruta": ["ID1.wav"] * 3 + ["ID2.wav"] * 2,
        "frame": [0, 1, 2, 0, 1],
        "f0": [1.0, NAN, 3.0, NAN, NAN],
        "jitter": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    prosody = pd.DataFrame({
        "id": ["ID1", "ID2"], "ruta": ["ID1.wav", "ID2.wav"], "frame": [0, 0],
        "energia": [1.0, 2.0],
    })
    return {"phonation": phonation, "prosody": prosody}


# --- limpiar_nan: estáticas -------------------------------------------------

def test_elimina_columnas_todo_nan_y_filas_vacias_de_todas_las_dimensiones(estaticas):
    r = limpiar_nan(estaticas, estrategia="media", verbose=False)
    limpio = r["caracteristicas"]

    assert r["columnas_eliminadas"] == {"phonation": ["vacia"], "prosody": []}
    assert "vacia" not in limpio["phonation"].columns
    # ID3 falló en phonation -> sale también de prosody
    assert list(limpio["phonation"]["id"]) == list(limpio["prosody"]["id"]) == ["ID1", "ID2", "ID4"]
    assert r["ids_eliminados"] == ["ID3"]


@pytest.mark.parametrize("estrategia, esperado", [
    ("media", 30.0),     # media de 10 y 50 (ID3 ya se eliminó)
    ("mediana", 30.0),
    ("moda", 10.0),      # empate: pandas devuelve el menor
    ("cero", 0.0),
])
def test_imputacion(estaticas, estrategia, esperado):
    r = limpiar_nan(estaticas, estrategia=estrategia, verbose=False)
    prosody = r["caracteristicas"]["prosody"]
    assert prosody.loc[prosody["id"] == "ID2", "energia"].item() == esperado
    assert r["estrategia"] == estrategia
    assert not any(df.isna().any().any() for df in r["caracteristicas"].values())


def test_eliminar_quita_al_participante_de_todas_las_dimensiones(estaticas):
    r = limpiar_nan(estaticas, estrategia="eliminar", verbose=False)
    for df in r["caracteristicas"].values():
        assert list(df["id"]) == ["ID1", "ID4"]
    assert r["ids_eliminados"] == ["ID2", "ID3"]


def test_sin_eliminar_filas_vacias_se_imputan(estaticas):
    r = limpiar_nan(estaticas, estrategia="cero", eliminar_filas_vacias=False, verbose=False)
    assert len(r["caracteristicas"]["phonation"]) == 4


def test_sin_nan_no_pregunta(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("no debía preguntar"))
    df = pd.DataFrame({"id": ["ID1", "ID2"], "f0": [1.0, 2.0]})
    r = limpiar_nan(df, verbose=False)
    assert r["estrategia"] == "ninguna"


def test_pregunta_hasta_recibir_una_opcion_valida(estaticas, monkeypatch):
    respuestas = iter(["promedio", "  Mediana "])
    monkeypatch.setattr("builtins.input", lambda _: next(respuestas))
    assert limpiar_nan(estaticas, verbose=False)["estrategia"] == "mediana"


def test_informe_en_consola(estaticas, capsys):
    limpiar_nan(estaticas, estrategia="media", nombre="estáticas")
    salida = capsys.readouterr().out
    assert "[estáticas] prosody: 1 de 3 filas con NaN" in salida
    assert "1 de 3 participantes tienen algún NaN (con 'eliminar' quedarían 2)" in salida


def test_dataframe_entra_dataframe_sale(estaticas):
    r = limpiar_nan(estaticas["phonation"], estrategia="media", verbose=False)
    assert isinstance(r["caracteristicas"], pd.DataFrame)
    assert r["columnas_eliminadas"] == ["vacia"]


def test_no_modifica_la_entrada(estaticas):
    copia = estaticas["prosody"].copy()
    limpiar_nan(estaticas, estrategia="cero", verbose=False)
    pd.testing.assert_frame_equal(estaticas["prosody"], copia)


def test_estrategia_invalida(estaticas):
    with pytest.raises(ValueError):
        limpiar_nan(estaticas, estrategia="promedio")


# --- limpiar_nan: dinámicas -------------------------------------------------

def test_dinamicas_eliminar_quita_frames_y_participantes_sin_frames(dinamicas):
    r = limpiar_nan(dinamicas, estrategia="eliminar", verbose=False)
    phonation = r["caracteristicas"]["phonation"]
    # ID1 pierde su frame con NaN; ID2 se queda sin frames -> sale de todas las dimensiones
    assert list(phonation["frame"]) == [0, 2] and set(phonation["id"]) == {"ID1"}
    assert list(r["caracteristicas"]["prosody"]["id"]) == ["ID1"]
    assert r["ids_eliminados"] == ["ID2"]


def test_dinamicas_imputacion(dinamicas):
    r = limpiar_nan(dinamicas, estrategia="media", verbose=False)
    phonation = r["caracteristicas"]["phonation"]
    # Los frames de ID2 no son todo-NaN (tienen jitter): se imputan con la media de f0 = 2.0
    assert phonation["f0"].tolist() == [1.0, 2.0, 3.0, 2.0, 2.0]


# --- pkl --------------------------------------------------------------------

def test_guardar_y_cargar_pkl(estaticas, tmp_path):
    ruta = guardar_pkl(estaticas, tmp_path / "sub" / "c.pkl")
    cargado = cargar_pkl(ruta)
    assert set(cargado) == {"phonation", "prosody"}
    pd.testing.assert_frame_equal(cargado["prosody"], estaticas["prosody"])


def test_las_funciones_aceptan_ruta_a_pkl(estaticas, tmp_path):
    ruta = guardar_pkl(estaticas, tmp_path / "c.pkl")
    assert obtener_ids(ruta) == ["ID1", "ID2", "ID3", "ID4"]
    assert set(limpiar_nan(ruta, estrategia="cero", verbose=False)["caracteristicas"]) == {
        "phonation", "prosody"}


# --- obtener_ids ------------------------------------------------------------

def test_obtener_ids_despues_de_limpiar(estaticas, tmp_path):
    limpio = limpiar_nan(estaticas, estrategia="eliminar", verbose=False)["caracteristicas"]
    ruta = tmp_path / "ids.txt"
    assert obtener_ids(limpio, ruta_txt=ruta) == ["ID1", "ID4"]
    assert leer_ids(ruta) == ["ID1", "ID4"]       # el txt se puede volver a leer


def test_obtener_ids_solo_los_comunes_a_todas_las_dimensiones(estaticas):
    estaticas["prosody"] = estaticas["prosody"][estaticas["prosody"]["id"] != "ID4"]
    assert obtener_ids(estaticas) == ["ID1", "ID2", "ID3"]


def test_obtener_ids_dinamicas_sin_repetir(dinamicas):
    assert obtener_ids(dinamicas) == ["ID1", "ID2"]


# --- seleccionar_caracteristicas --------------------------------------------

def test_seleccionar_con_lista(estaticas):
    r = seleccionar_caracteristicas(estaticas, ["ID4", "ID1", "ID9"], verbose=False)
    for df in r["caracteristicas"].values():
        assert list(df["id"]) == ["ID1", "ID4"]           # conserva el orden de la matriz
    assert r["ids_no_encontrados"] == ["ID9"]


def test_seleccionar_con_csv(estaticas, tmp_path):
    ruta = tmp_path / "ids.csv"
    ruta.write_text("id;grupo\nID2;GG\nID3;GU\n")
    r = seleccionar_caracteristicas(estaticas, ruta, columna="id", verbose=False)
    assert list(r["caracteristicas"]["prosody"]["id"]) == ["ID2", "ID3"]


def test_seleccionar_con_txt_y_dataframe(estaticas, tmp_path):
    ruta = tmp_path / "ids.txt"
    ruta.write_text("ID3\n")
    r = seleccionar_caracteristicas(estaticas["phonation"], ruta, verbose=False)
    assert isinstance(r["caracteristicas"], pd.DataFrame) and len(r["caracteristicas"]) == 1


def test_seleccionar_dinamicas_conserva_todos_los_frames(dinamicas):
    r = seleccionar_caracteristicas(dinamicas, ["ID1"], verbose=False)
    assert len(r["caracteristicas"]["phonation"]) == 3


def test_seleccionar_ids_numericos_en_la_matriz():
    df = pd.DataFrame({"id": [1, 2, 3], "f0": [1.0, 2.0, 3.0]})
    r = seleccionar_caracteristicas(df, ["2"], verbose=False)
    assert r["caracteristicas"]["f0"].tolist() == [2.0]


def test_seleccionar_sin_coincidencias(estaticas):
    with pytest.raises(ValueError):
        seleccionar_caracteristicas(estaticas, ["001", "002"], verbose=False)


# --- obtener_clases ---------------------------------------------------------

@pytest.fixture
def ids_grupos():
    return pd.Series(["GG_001", "GU_002", "GG_003", "CTRL_4"], index=[10, 11, 12, 13])


def test_obtener_clases_lista(ids_grupos):
    y = obtener_clases(ids_grupos[:3], ["GG", "GU"])
    assert y.tolist() == ["GG", "GU", "GG"]
    assert y.name == "clase"
    assert list(y.index) == [10, 11, 12]


def test_obtener_clases_dict_y_mayusculas(ids_grupos):
    y = obtener_clases(ids_grupos, {0: "gg", 1: "gu", 2: "ctrl"}, ignorar_mayusculas=True)
    assert y.tolist() == [0, 1, 0, 2]


def test_obtener_clases_sin_clase(ids_grupos):
    with pytest.raises(ValueError, match="CTRL_4"):
        obtener_clases(ids_grupos, ["GG", "GU"])
    y = obtener_clases(ids_grupos, ["GG", "GU"], sin_clase="nan")
    assert y.isna().tolist() == [False, False, False, True]


def test_obtener_clases_ambiguos(ids_grupos):
    with pytest.raises(ValueError, match="más de una clase"):
        obtener_clases(ids_grupos, {"a": "G", "b": "GG"})


def test_obtener_clases_regex_y_literal():
    ids = ["GG.1", "XGG2", "GGx3"]
    assert obtener_clases(ids, {"gg": "^GG"}, regex=True, sin_clase="nan").isna().tolist() == [False, True, False]
    # literal: el punto no es comodín
    assert obtener_clases(ids, {"p": "GG."}, sin_clase="nan").notna().tolist() == [True, False, False]


def test_obtener_clases_argumentos_invalidos(ids_grupos):
    with pytest.raises(ValueError):
        obtener_clases(ids_grupos, [])
    with pytest.raises(ValueError):
        obtener_clases(ids_grupos, ["GG"], sin_clase="otra")
