"""
Pruebas de gita.audio.disvoice con un paquete `disvoice` FALSO.

El DisVoice real no se puede instalar en GitHub Actions (TensorFlow, phonet,
Praat...). Estas pruebas verifican el circuito completo
(lista -> proceso aparte -> CSV -> DataFrame), no los valores de DisVoice.
"""

import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from gita.audio.disvoice import caracterizar_disvoice

DISVOICE_FALSO = {
    "__init__.py": "",
    "phonation.py": """
        import os
        import numpy as np

        class Phonation:
            head_st = ["avg DF0", "std DF0", "avg Jitter"]

            def extract_features_file(self, audio, static=True, plots=False, fmt="npy"):
                nombre = os.path.basename(audio)
                if "falla" in nombre:
                    raise RuntimeError("audio dañado")
                if "inf" in nombre:
                    return np.array([np.inf, 1.0, 2.0])
                return np.array([1.0, 2.0, 3.0])
    """,
    "phonological.py": """
        import numpy as np

        class Phonological:
            head_st = ["vocalic", "nasal"]

            def extract_features_file(self, audio, static=True, plots=False, fmt="npy"):
                return np.array([0.5, 0.25])
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


def test_extrae_phonation_desde_carpeta(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, verbose=False)
    df = r["caracteristicas"]["phonation"]

    assert list(df.columns) == ["id", "ruta", "avg DF0", "std DF0", "avg Jitter"]
    assert list(df["id"]) == ["ID001", "ID002_inf", "ID003_falla"]
    assert df.iloc[0, 2:].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(df.iloc[1]["avg DF0"])                 # inf -> NaN
    assert df.iloc[2, 2:].isna().all()                      # el que falla: fila en NaN
    assert len(r["fallidos"]) == 1 and "audio dañado" in r["fallidos"][0]["error"]


def test_acepta_lista_de_rutas(disvoice_falso, audios):
    r = caracterizar_disvoice([audios / "tarea1" / "ID001.wav"], verbose=False)
    assert len(r["caracteristicas"]["phonation"]) == 1


def test_varias_dimensiones_en_un_solo_proceso(disvoice_falso, audios):
    r = caracterizar_disvoice(audios, ["phonation", "phonological"], verbose=False)
    assert set(r["caracteristicas"]) == {"phonation", "phonological"}
    assert r["caracteristicas"]["phonological"]["nasal"].tolist() == [0.25] * 3


def test_guarda_y_reutiliza_csv(disvoice_falso, audios, tmp_path):
    salida = tmp_path / "caracteristicas"
    caracterizar_disvoice(audios, carpeta_salida=salida, verbose=False)
    assert (salida / "phonation.csv").exists()
    assert (salida / "errores_disvoice.csv").exists()

    # Segunda vez: se carga el CSV; ni siquiera necesita el Python de DisVoice
    r = caracterizar_disvoice(audios, carpeta_salida=salida, python_disvoice="/no/existe",
                              verbose=False)
    assert len(r["caracteristicas"]["phonation"]) == 3


def test_csv_existente_de_otra_lista(disvoice_falso, audios, tmp_path):
    salida = tmp_path / "caracteristicas"
    caracterizar_disvoice(audios, carpeta_salida=salida, verbose=False)
    with pytest.raises(ValueError):
        caracterizar_disvoice([audios / "tarea1" / "ID001.wav"], carpeta_salida=salida,
                              verbose=False)


def test_phonological_exige_16k(disvoice_falso, audios):
    crear_wav(audios / "tarea1" / "ID004.wav", fs=44100)
    with pytest.raises(ValueError, match="16 kHz"):
        caracterizar_disvoice(audios, "phonological", verbose=False)


def test_sin_python_de_disvoice(audios, monkeypatch):
    monkeypatch.delenv("GITA_DISVOICE_PYTHON", raising=False)
    with pytest.raises(ValueError, match="GITA_DISVOICE_PYTHON"):
        caracterizar_disvoice(audios, verbose=False)


def test_dimension_invalida(audios):
    with pytest.raises(ValueError):
        caracterizar_disvoice(audios, "fonacion", verbose=False)


def test_error_del_proceso_se_reporta(disvoice_falso, audios):
    # El disvoice falso no tiene 'prosody': el worker termina con error
    with pytest.raises(RuntimeError, match="DisVoice"):
        caracterizar_disvoice(audios, "prosody", verbose=False)
