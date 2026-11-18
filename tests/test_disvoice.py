"""
Pruebas de gita.audio.caracterizacion (DisVoice) con un paquete `disvoice` FALSO.

El DisVoice real no se puede instalar en GitHub Actions (TensorFlow, phonet,
Praat...). Estas pruebas verifican el circuito completo
(lista -> proceso aparte -> CSV -> DataFrame), no los valores de DisVoice.
"""

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from gita.audio.caracterizacion import caracterizar_disvoice
from gita.audio.caracterizacion import disvoice as modulo

DISVOICE_FALSO = {
    "__init__.py": "",
    "phonation.py": """
        import os
        import numpy as np
        import pandas as pd

        class Phonation:
            head_st = ["avg DF0", "std DF0", "avg Jitter"]

            def extract_features_file(self, audio, static=True, plots=False, fmt="npy"):
                nombre = os.path.basename(audio)
                if "falla" in nombre:
                    raise RuntimeError("audio dañado")
                if static:
                    if "inf" in nombre:
                        return np.array([np.inf, 1.0, 2.0])
                    return np.array([1.0, 2.0, 3.0])
                # Dinámico: DataFrame con columna 'time' (que se debe quitar), 4 frames
                return pd.DataFrame({"time": [0, 1, 2, 3], "DF0": [1.0, 2.0, np.inf, 4.0],
                                     "Jitter": [0.1, 0.2, 0.3, 0.4]})
    """,
    "phonological.py": """
        import numpy as np

        class Phonological:
            head_st = ["vocalic", "nasal"]
            head_dyn = ["vocalic", "nasal"]

            def extract_features_file(self, audio, static=True, plots=False, fmt="npy"):
                if static:
                    return np.array([0.5, 0.25])
                return np.ones((3, 2))   # dinámico como arreglo: usa head_dyn
    """,
}


@pytest.fixture
def disvoice_falso(tmp_path, monkeypatch):
    """Crea el paquete falso y lo pone en el PYTHONPATH del proceso del worker."""
    raiz = tmp_path / "paquetes"
    (raiz / "disvoice").mkdir(parents=True)
    for nombre, codigo in DISVOICE_FALSO.items():
        (raiz / "disvoice" / nombre).write_text(textwrap.dedent(codigo))
    monkeypatch.setenv("PYTHONPATH", str(raiz))
    monkeypatch.setenv("GITA_DISVOICE_PYTHON", sys.executable)
    monkeypatch.chdir(tmp_path)   # la carpeta "caracteristicas" por defecto queda aquí


def crear_wav(ruta, fs=16000):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(ruta), 0.3 * np.sin(np.linspace(0, 100, fs)), fs)


@pytest.fixture
def audios(tmp_path):
    carpeta = tmp_path / "audios"
    crear_wav(carpeta / "tarea1" / "ID001.wav")
    crear_wav(carpeta / "tarea1" / "ID002_inf.wav")
    crear_wav(carpeta / "tarea2" / "ID003_falla.wav")
    return carpeta


# --- estáticas --------------------------------------------------------------

def test_estaticas_desde_carpeta(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, verbose=False)
    df = r["estaticas"]["phonation"]

    assert r["dinamicas"] == {}
    assert list(df.columns) == ["id", "ruta", "avg DF0", "std DF0", "avg Jitter"]
    assert list(df["id"]) == ["ID001", "ID002_inf", "ID003_falla"]
    assert df.iloc[0, 2:].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(df.iloc[1]["avg DF0"])                 # inf -> NaN
    assert df.iloc[2, 2:].isna().all()                      # el que falla: fila en NaN
    assert len(r["fallidos"]) == 1
    assert r["fallidos"][0]["tipo"] == "estaticas" and "audio dañado" in r["fallidos"][0]["error"]


def test_acepta_lista_de_rutas(disvoice_falso, audios):
    r = caracterizar_disvoice([audios / "tarea1" / "ID001.wav"], verbose=False)
    assert len(r["estaticas"]["phonation"]) == 1


# --- dinámicas --------------------------------------------------------------

def test_dinamicas(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, estaticas=False, dinamicas=True, verbose=False)
    df = r["dinamicas"]["phonation"]

    assert r["estaticas"] == {}
    assert list(df.columns) == ["id", "ruta", "frame", "DF0", "Jitter"]   # sin 'time'
    assert df.groupby("id").size().to_dict() == {"ID001": 4, "ID002_inf": 4}  # el que falla no aporta
    assert list(df["frame"][:4]) == [0, 1, 2, 3]
    assert df["DF0"].isna().sum() == 2                       # un inf por audio -> NaN
    assert [f["tipo"] for f in r["fallidos"]] == ["dinamicas"]


def test_dinamicas_como_arreglo_usa_head_dyn(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, "phonological", estaticas=False, dinamicas=True,
                              verbose=False)
    assert list(r["dinamicas"]["phonological"].columns) == ["id", "ruta", "frame", "vocalic", "nasal"]


def test_estaticas_y_dinamicas_a_la_vez(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, ["phonation", "phonological"], dinamicas=True, verbose=False)
    assert set(r["estaticas"]) == set(r["dinamicas"]) == {"phonation", "phonological"}
    assert r["estaticas"]["phonological"]["nasal"].tolist() == [0.25] * 3


# --- carpeta de salida ------------------------------------------------------

def test_carpeta_por_defecto_en_directorio_de_trabajo(disvoice_falso, audios, tmp_path):
    caracterizar_disvoice(audios, dinamicas=True, verbose=False)
    carpeta = tmp_path / "caracteristicas"
    assert (carpeta / "phonation.csv").exists()
    assert (carpeta / "phonation_dyn.csv").exists()
    assert (carpeta / "errores_disvoice.csv").exists()


def test_sin_carpeta_no_guarda(disvoice_falso, audios, tmp_path):
    caracterizar_disvoice(audios, carpeta_salida=None, verbose=False)
    assert not (tmp_path / "caracteristicas").exists()


def test_reutiliza_csv_existentes(disvoice_falso, audios):
    caracterizar_disvoice(audios, dinamicas=True, verbose=False)

    # Segunda vez: todo se carga; ni siquiera necesita el Python de DisVoice.
    # En dinámicas el audio que falló no tiene frames, pero está en el registro de errores.
    r = caracterizar_disvoice(audios, dinamicas=True, python_disvoice="/no/existe", verbose=False)
    assert len(r["estaticas"]["phonation"]) == 3
    assert len(r["dinamicas"]["phonation"]) == 8
    assert len(r["fallidos"]) == 2


def test_sobrescribir_rehace_y_actualiza_errores(disvoice_falso, audios, tmp_path):
    caracterizar_disvoice(audios, verbose=False)
    r = caracterizar_disvoice(audios, sobrescribir=True, verbose=False)
    assert len(r["fallidos"]) == 1   # el error no queda duplicado en el registro


def test_csv_existente_de_otra_lista(disvoice_falso, audios):
    caracterizar_disvoice(audios, verbose=False)
    with pytest.raises(ValueError, match="otra lista"):
        caracterizar_disvoice([audios / "tarea1" / "ID001.wav"], verbose=False)


# --- validaciones -----------------------------------------------------------

def test_phonological_exige_16k(disvoice_falso, audios):
    crear_wav(audios / "tarea1" / "ID004.wav", fs=44100)
    with pytest.raises(ValueError, match="16 kHz"):
        caracterizar_disvoice(audios, "phonological", verbose=False)


def test_ningun_tipo(audios):
    with pytest.raises(ValueError):
        caracterizar_disvoice(audios, estaticas=False, dinamicas=False, verbose=False)


def test_dimension_invalida(audios):
    with pytest.raises(ValueError):
        caracterizar_disvoice(audios, "fonacion", verbose=False)


def test_error_del_proceso_se_reporta(disvoice_falso, audios):
    # El disvoice falso no tiene 'prosody': el worker termina con error
    with pytest.raises(RuntimeError, match="DisVoice"):
        caracterizar_disvoice(audios, "prosody", verbose=False)


# --- Python de DisVoice -----------------------------------------------------

def test_detecta_entorno_conda_hermano(tmp_path, monkeypatch):
    monkeypatch.delenv("GITA_DISVOICE_PYTHON", raising=False)
    envs = tmp_path / "miniconda3" / "envs"
    python = envs / "disvoice" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    (envs / "portables").mkdir()
    monkeypatch.setattr(modulo.sys, "prefix", str(envs / "portables"))
    assert modulo._python_disvoice(None) == str(python)


def test_parametro_tiene_prioridad(monkeypatch):
    monkeypatch.setenv("GITA_DISVOICE_PYTHON", "/otro/python")
    assert modulo._python_disvoice(sys.executable) == sys.executable


def test_sin_python_de_disvoice(tmp_path, monkeypatch):
    monkeypatch.delenv("GITA_DISVOICE_PYTHON", raising=False)
    monkeypatch.setattr(modulo.sys, "prefix", str(tmp_path / "no" / "existe"))
    with pytest.raises(ValueError, match="GITA_DISVOICE_PYTHON"):
        modulo._python_disvoice(None)


def test_filtro_de_ruido():
    assert modulo._RUIDO.match("5/5 [==============================] - 0s 13ms/step\n")
    assert modulo._RUIDO.match("### WARNING, path does not exist: KALDI_ROOT=/mnt\n")
    assert modulo._RUIDO.match("################\n")
    assert modulo._RUIDO.match("\n")
    assert not modulo._RUIDO.match("[phonation estáticas] 1/3 ID001.wav\n")
