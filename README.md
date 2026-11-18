# gita

Herramientas de procesamiento de señales y aprendizaje automático del GITA Lab
(Universidad de Antioquia). La librería crece a medida que se necesitan funciones.

```python
import gita
print(gita.__version__)
```

## Instalación

Desde cualquier computador con Python ≥ 3.8 y git:

```bash
pip install git+https://github.com/Santiago6323/gita.git
```

Una versión fija (recomendado en experimentos, para que los resultados sean reproducibles):

```bash
pip install git+https://github.com/Santiago6323/gita.git@v0.2.0
```

Actualizar a la última versión:

```bash
pip install --upgrade --force-reinstall git+https://github.com/Santiago6323/gita.git
```

En el `requirements.txt` de un proyecto:

```
gita-lab @ git+https://github.com/Santiago6323/gita.git@v0.2.0
```

> El paquete se instala como `gita-lab` (`pip uninstall gita-lab`) y se importa como `gita`.
> No uses `pip install gita`: ese nombre en PyPI pertenece a otra herramienta.

## Módulos

### `gita.audio` — preprocesamiento

Lleva cualquier audio a mono, sin DC, 16 kHz y normalizado por pico. El orden es
mono → eliminar DC → remuestreo → normalización: el DC se quita antes de remuestrear
porque con offset el filtro del remuestreo produce transitorios en los bordes, y se
normaliza al final porque ese filtro cambia el pico. Dependencias: `numpy`, `scipy`, `soundfile` (incluidas en la
instalación base).

```python
from gita.audio import preprocesar, preprocesar_archivo, preprocesar_carpeta

# Un archivo: lee, procesa y guarda (crea las carpetas de salida si no existen)
preprocesar_archivo("original.wav", "procesado/original.wav")

# Un arreglo en memoria (1D mono o 2D (n, canales), como lo da soundfile.read)
senal_16k, fs = preprocesar(senal, fs_original)

# Una carpeta completa (mantiene subcarpetas; .flac -> .wav; salta los ya procesados)
resultado = preprocesar_carpeta("audios_originales/", "audios_16k/")
print(resultado["fallidos"])
```

| Función | Qué hace |
|---|---|
| `a_mono(senal)` | Promedia los canales: (n, canales) → (n,) |
| `eliminar_dc(senal)` | Resta la media de cada canal: y = x − media(x) |
| `remuestrear(senal, fs_origen, fs_destino=16000)` | Remuestreo polifásico con antialiasing (`scipy.signal.resample_poly`) |
| `normalizar_pico(senal, pico=1.0)` | Escala para que max(\|x\|) = pico; el silencio se deja igual |
| `preprocesar(senal, fs, fs_destino=16000, mono=True, quitar_dc=True, normalizar=True, pico=1.0)` | Las cuatro anteriores en orden; devuelve `(senal, fs_destino)` |
| `preprocesar_archivo(entrada, salida, ..., subtipo="PCM_16")` | Lo mismo de archivo a archivo; devuelve `(salida, fs_destino)` |
| `preprocesar_carpeta(entrada, salida, extensiones=(".wav", ".flac"), recursivo=True, sobrescribir=False, ...)` | Lo mismo para toda una carpeta, manteniendo subcarpetas; devuelve un dict con `procesados`, `saltados` y `fallidos` |

### `gita.datos` — selección de archivos por IDs

Selecciona de una base de datos solo los archivos de una lista de IDs. El ID de un
archivo es su nombre sin extensión (`tarea1/ID001.wav` → `ID001`). No lee el contenido
de los archivos, así que sirve para audio, IMU, EEG, etc. Sin dependencias extra.

```python
from gita.datos import leer_ids, seleccionar_por_ids, crear_enlaces

ids = leer_ids("ids.csv", columna="id")          # o leer_ids("ids.txt"): uno por línea
r = seleccionar_por_ids("audios_16k/", ids)       # busca en todas las subcarpetas (tareas)
print(r["ids_no_encontrados"])                    # IDs de la lista sin ningún archivo

# Carpeta con solo los seleccionados, sin copiar datos (enlaces simbólicos)
crear_enlaces(r["seleccionados"], "audios_16k/", "seleccionados/")

# Las tres en una sola llamada
from gita.datos import seleccionar_datos
r = seleccionar_datos("audios_16k/", "ids.csv", "seleccionados/", columna="id")
```

| Función | Qué hace |
|---|---|
| `leer_ids(ruta, columna=None)` | IDs de un `.txt` (uno por línea) o de una columna de un `.csv`/`.tsv` (separador `,` `;` o tabulación detectado solo); quita espacios y duplicados |
| `seleccionar_por_ids(carpeta, ids, extensiones=(".wav",), recursivo=True)` | Rutas de los archivos cuyo nombre está en `ids`; devuelve un dict con `seleccionados` e `ids_no_encontrados`. Coincidencia exacta (`001` ≠ `1`, `id001` ≠ `ID001`) |
| `crear_enlaces(rutas, carpeta_base, carpeta_destino)` | Crea enlaces simbólicos manteniendo la estructura relativa a `carpeta_base`. La carpeta destino queda exactamente con la selección actual (borra los enlaces de una ejecución anterior, nunca los originales; si encuentra archivos reales, no toca nada) |
| `seleccionar_datos(carpeta, ids, carpeta_destino, columna=None, extensiones=(".wav",))` | Las tres anteriores en una llamada; `ids` es un archivo o una lista. Si nada coincide, se detiene sin tocar la carpeta destino. Devuelve `seleccionados`, `enlaces` e `ids_no_encontrados` |

## Estructura

```
src/gita/        código de la librería (un subpaquete por área: audio, modelos, ...)
tests/           pruebas (pytest)
pyproject.toml   metadatos, versión y dependencias
```

## Desarrollo

```bash
git clone https://github.com/Santiago6323/gita.git
cd gita
pip install -e ".[dev]"   # editable: los cambios en src/ se ven sin reinstalar
pytest
```

Para publicar una versión nueva:

1. Agregar la función en el subpaquete que corresponda, con su prueba en `tests/`
   y sus dependencias en `pyproject.toml`.
2. Subir `version` en `pyproject.toml` (p. ej. `0.0.1` → `0.1.0`).
3. `git commit`, `git tag v0.1.0`, `git push && git push --tags`.

## Licencia

MIT
