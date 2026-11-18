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

### `gita.audio.disvoice` — caracterización con DisVoice

DisVoice necesita su propio entorno de conda (TensorFlow, phonet...), así que gita no
lo importa: lo ejecuta como un proceso aparte con el Python de ese entorno. No hay que
instalar gita en el entorno de DisVoice. Dependencia en el entorno de gita: `pandas`.

Indica una sola vez dónde está el Python de DisVoice (p. ej. en `~/.bashrc`):

```bash
export GITA_DISVOICE_PYTHON=~/miniconda3/envs/disvoice/bin/python
```

o pásalo como argumento: `caracterizar_disvoice(..., python_disvoice="...")`.

```python
from gita.audio.disvoice import caracterizar_disvoice

r = caracterizar_disvoice(
    "audios_16k/",                      # carpeta o lista de rutas (p. ej. r["seleccionados"])
    ["phonation", "prosody"],           # una o varias dimensiones
    carpeta_salida="caracteristicas/",  # opcional: guarda phonation.csv, prosody.csv, ...
)
df = r["caracteristicas"]["phonation"]  # columnas: id, ruta, características de DisVoice
print(r["fallidos"])                    # audios que fallaron (su fila queda en NaN)
```

| Función | Qué hace |
|---|---|
| `caracterizar_disvoice(entrada, dimensiones=("phonation",), python_disvoice=None, carpeta_salida=None, sobrescribir=False)` | Características estáticas (un vector por audio) de `phonation`, `prosody`, `articulation`, `phonological` y/o `glottal`, todas en un solo proceso. Devuelve `{"caracteristicas": {dimension: DataFrame}, "fallidos": [...]}` |

- Columnas: `id` (nombre sin extensión), `ruta` (distingue el mismo `id` en tareas
  distintas) y las de `head_st` de DisVoice.
- Los NaN de DisVoice se dejan como NaN; los ±inf se convierten a NaN. Un audio que falla
  queda con su fila en NaN y su error en `fallidos` (y en `errores_disvoice.csv`).
- Con `carpeta_salida`, una dimensión cuyo CSV ya existe se carga en vez de extraerse de
  nuevo (`sobrescribir=True` para rehacerla).
- `phonological` exige audios a 16 kHz (con otra frecuencia phonet devuelve NaN): si
  alguno no lo está, se detiene antes de extraer.

### `gita.datos` — selección de archivos por IDs

Selecciona de una base de datos solo los archivos de una lista de IDs. El ID de un
archivo es su nombre sin extensión (`tarea1/ID001.wav` → `ID001`). No lee el contenido
de los archivos, así que sirve para audio, IMU, EEG, etc. Sin dependencias extra.

La selección es una **lista de rutas**: no se copia nada. Las funciones que procesan
varios archivos (p. ej. la caracterización) aceptan una carpeta **o** una lista de
rutas, gracias a `listar_archivos`.

```python
from gita.datos import seleccionar_datos

r = seleccionar_datos("audios_16k/", "ids.csv", columna="id")   # o una lista: ["ID001", ...]
print(r["ids_no_encontrados"])        # IDs de la lista sin ningún archivo

caracterizar("audios_16k/")           # todos los audios
caracterizar(r["seleccionados"])      # solo los de la lista
```

| Función | Qué hace |
|---|---|
| `listar_archivos(entrada, extensiones=(".wav",), recursivo=True)` | Normaliza la entrada: carpeta → sus archivos; archivo → `[archivo]`; lista → lista de `Path` (verifica que existan) |
| `leer_ids(ruta, columna=None)` | IDs de un `.txt` (uno por línea) o de una columna de un `.csv`/`.tsv` (separador `,` `;` o tabulación detectado solo); quita espacios y duplicados |
| `seleccionar_por_ids(carpeta, ids, extensiones=(".wav",), recursivo=True)` | Rutas de los archivos cuyo nombre está en `ids`; devuelve un dict con `seleccionados` e `ids_no_encontrados`. Coincidencia exacta (`001` ≠ `1`, `id001` ≠ `ID001`) |
| `seleccionar_datos(carpeta, ids, carpeta_destino=None, columna=None, ...)` | `leer_ids` + `seleccionar_por_ids` en una llamada; `ids` es un archivo o una lista. Si nada coincide, lanza un error. Con `carpeta_destino`, además crea una carpeta de enlaces |
| `crear_enlaces(rutas, carpeta_base, carpeta_destino)` | Solo para herramientas externas que exigen una carpeta: crea enlaces simbólicos (sin copiar datos) manteniendo la estructura. La carpeta destino queda exactamente con la selección actual y nunca se tocan archivos reales |

## Estructura

```
src/gita/        código de la librería (un subpaquete por área: audio, modelos, ...)
src/gita/audio/_workers/  scripts que se ejecutan con el Python de otros entornos (DisVoice)
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
