import numpy as np
import pytest
import soundfile as sf

from gita.audio import preprocesar_carpeta


def crear_audio(ruta, fs=44100, canales=2):
    ruta.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(fs) / fs
    x = 0.3 * np.sin(2 * np.pi * 440 * t)
    sf.write(str(ruta), np.column_stack([x] * canales), fs)


@pytest.fixture
def carpetas(tmp_path):
    entrada = tmp_path / "entrada"
    crear_audio(entrada / "a.wav")
    crear_audio(entrada / "sujeto01" / "b.flac", fs=48000)
    (entrada / "notas.txt").write_text("no es audio")
    return entrada, tmp_path / "salida"


def test_procesa_y_mantiene_estructura(carpetas):
    entrada, salida = carpetas
    r = preprocesar_carpeta(entrada, salida, verbose=False)

    assert len(r["procesados"]) == 2 and not r["fallidos"]
    for ruta in [salida / "a.wav", salida / "sujeto01" / "b.wav"]:  # .flac -> .wav
        info = sf.info(str(ruta))
        assert info.samplerate == 16000 and info.channels == 1


def test_no_recursivo(carpetas):
    entrada, salida = carpetas
    r = preprocesar_carpeta(entrada, salida, recursivo=False, verbose=False)
    assert len(r["procesados"]) == 1


def test_salta_existentes_y_sobrescribe(carpetas):
    entrada, salida = carpetas
    preprocesar_carpeta(entrada, salida, verbose=False)
    r = preprocesar_carpeta(entrada, salida, verbose=False)
    assert len(r["saltados"]) == 2 and not r["procesados"]
    r = preprocesar_carpeta(entrada, salida, sobrescribir=True, verbose=False)
    assert len(r["procesados"]) == 2


def test_archivo_danado_no_detiene_lote(carpetas):
    entrada, salida = carpetas
    (entrada / "danado.wav").write_bytes(b"esto no es un wav")
    r = preprocesar_carpeta(entrada, salida, verbose=False)
    assert len(r["procesados"]) == 2 and len(r["fallidos"]) == 1


def test_salida_dentro_de_entrada(carpetas):
    entrada, _ = carpetas
    with pytest.raises(ValueError):
        preprocesar_carpeta(entrada, entrada / "salida", verbose=False)


def test_entrada_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocesar_carpeta(tmp_path / "no_existe", tmp_path / "salida")
