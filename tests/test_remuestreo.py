import numpy as np
import pytest
import soundfile as sf

from gita.audio import (
    a_mono,
    normalizar_pico,
    preprocesar,
    preprocesar_archivo,
    remuestrear,
)


def seno(f, fs, duracion=1.0, amplitud=0.5):
    t = np.arange(int(fs * duracion)) / fs
    return amplitud * np.sin(2 * np.pi * f * t)


def frecuencia_dominante(senal, fs):
    espectro = np.abs(np.fft.rfft(senal))
    return np.fft.rfftfreq(len(senal), 1 / fs)[np.argmax(espectro)]


# --- a_mono -----------------------------------------------------------------

def test_a_mono_promedia_canales():
    estereo = np.column_stack([np.ones(10), np.zeros(10)])
    assert np.allclose(a_mono(estereo), 0.5)


def test_a_mono_deja_igual_senal_1d():
    x = seno(100, 8000)
    assert np.array_equal(a_mono(x), x)


def test_a_mono_rechaza_3d():
    with pytest.raises(ValueError):
        a_mono(np.zeros((2, 2, 2)))


# --- remuestrear ------------------------------------------------------------

@pytest.mark.parametrize("fs_origen", [8000, 22050, 44100, 48000])
def test_remuestrear_longitud_y_frecuencia(fs_origen):
    x = seno(440, fs_origen)
    y = remuestrear(x, fs_origen, 16000)
    assert len(y) == 16000  # 1 s a 16 kHz
    assert abs(frecuencia_dominante(y, 16000) - 440) <= 1


def test_remuestrear_misma_fs_devuelve_copia():
    x = seno(100, 16000)
    y = remuestrear(x, 16000, 16000)
    assert np.array_equal(x, y) and y is not x


def test_remuestrear_multicanal():
    x = np.column_stack([seno(440, 44100), seno(880, 44100)])
    assert remuestrear(x, 44100, 16000).shape == (16000, 2)


def test_remuestrear_fs_invalida():
    with pytest.raises(ValueError):
        remuestrear(np.zeros(10), 0, 16000)


# --- normalizar_pico --------------------------------------------------------

def test_normalizar_pico():
    y = normalizar_pico(seno(100, 8000, amplitud=0.2), pico=0.9)
    assert np.isclose(np.max(np.abs(y)), 0.9)


def test_normalizar_silencio_no_divide_por_cero():
    y = normalizar_pico(np.zeros(100))
    assert np.all(y == 0) and np.all(np.isfinite(y))


# --- preprocesar ------------------------------------------------------------

def test_preprocesar_estereo_44k():
    x = np.column_stack([seno(440, 44100, amplitud=0.3)] * 2)
    y, fs = preprocesar(x, 44100)
    assert fs == 16000
    assert y.ndim == 1 and len(y) == 16000
    assert np.isclose(np.max(np.abs(y)), 1.0)


def test_preprocesar_sin_mono_ni_normalizar():
    x = np.column_stack([seno(440, 48000, amplitud=0.3)] * 2)
    y, _ = preprocesar(x, 48000, mono=False, normalizar=False)
    assert y.shape == (16000, 2)
    assert np.max(np.abs(y)) < 0.35


# --- preprocesar_archivo ----------------------------------------------------

def test_preprocesar_archivo(tmp_path):
    entrada = tmp_path / "original.wav"
    salida = tmp_path / "sub" / "carpeta" / "procesado.wav"  # carpetas inexistentes
    sf.write(str(entrada), np.column_stack([seno(440, 44100)] * 2), 44100)

    ruta, fs = preprocesar_archivo(str(entrada), str(salida))

    info = sf.info(ruta)
    assert fs == 16000 and info.samplerate == 16000
    assert info.channels == 1 and info.subtype == "PCM_16"
    y, _ = sf.read(ruta)
    assert np.max(np.abs(y)) > 0.99
