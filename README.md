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

### `gita.audio.caracterizacion` — caracterización (DisVoice)

Todas las funciones de caracterización aceptan como entrada **una carpeta o una lista
de rutas** (p. ej. `seleccionar_datos(...)["seleccionados"]`).

**DisVoice** necesita su propio entorno de conda (TensorFlow, phonet...), así que gita no
lo importa: lo ejecuta como un proceso aparte con el Python de ese entorno. No hay que
instalar gita en el entorno de DisVoice. Dependencia en el entorno de gita: `pandas`.

El Python de DisVoice se busca en este orden:

1. el parámetro `python_disvoice="..."`
2. la variable de entorno `GITA_DISVOICE_PYTHON`
3. un entorno de conda llamado `disvoice` junto al entorno actual
   (p. ej. `envs/portables` → `envs/disvoice`): sin configurar nada

```python
from gita.audio.caracterizacion import caracterizar_disvoice

r = caracterizar_disvoice(
    "audios_16k/",                 # carpeta o lista de rutas
    ["phonation", "prosody"],      # una o varias dimensiones
    estaticas=True,                # un vector por audio
    dinamicas=True,                # un vector por frame (ambos modos en el mismo proceso)
    nan="preguntar",               # o "media", "mediana", "moda", "cero", "eliminar"
)
r["estaticas"]["phonation"]        # limpio. Columnas: id, ruta, características
r["dinamicas"]["phonation"]        # limpio. Columnas: id, ruta, frame, características
r["crudas"]["estaticas"]           # sin limpiar
r["limpieza"]                      # columnas e ids eliminados, estrategia aplicada
r["fallidos"]                      # audios que fallaron (dimension, tipo, ruta, error)
```

Después de extraer, limpia los NaN con `gita.caracteristicas.limpiar_nan` (por separado
para estáticas y dinámicas): borra las columnas todo-NaN y los audios fallidos, informa
cuántas filas tienen NaN y, con `nan="preguntar"`, pregunta en la consola qué hacer.
Un participante eliminado sale de todas las dimensiones.

Por defecto guarda todo en `caracteristicas/`, **relativa al directorio de trabajo** (en
Spyder con `%runfile --wdir`, la carpeta del script):

```
caracteristicas/
  phonation.csv            estáticas crudas (sin limpiar)
  phonation_dyn.csv        dinámicas crudas
  prosody.csv ...
  disvoice_estaticas.pkl   {dimension: DataFrame} limpio
  disvoice_dinamicas.pkl   {dimension: DataFrame} limpio
  errores_disvoice.csv     registro de audios que fallaron (con traceback)
```

| Función | Qué hace |
|---|---|
| `caracterizar_disvoice(entrada, dimensiones=("phonation",), estaticas=True, dinamicas=False, carpeta_salida="caracteristicas", sobrescribir=False, nan="preguntar", eliminar_filas_vacias=True, python_disvoice=None)` | Características de `phonation`, `prosody`, `articulation`, `phonological` y/o `glottal`, estáticas y/o dinámicas, en un solo proceso; las limpia y guarda un pkl por tipo. Devuelve `{"estaticas", "dinamicas", "crudas", "limpieza", "fallidos"}` |

- `id` es el nombre sin extensión y `ruta` distingue el mismo `id` en tareas distintas.
- Los NaN de DisVoice se conservan; los ±inf se convierten a NaN. Un audio que falla queda
  con su fila en NaN (estáticas) o sin frames (dinámicas), y su error en `fallidos`.
- Un CSV crudo que ya existe se carga en vez de extraerse de nuevo (`sobrescribir=True`
  para rehacerlo); la limpieza y el pkl se rehacen siempre. Si el CSV es de otra lista
  de audios, se detiene con un error.
  `carpeta_salida=None` no guarda nada.
- `phonological` exige audios a 16 kHz (con otra frecuencia phonet devuelve NaN): si
  alguno no lo está, se detiene antes de extraer.
- Dinámicas: qué es un frame depende de la dimensión (phonation: ventana sonora;
  articulation: onset sordo→sonoro; prosody: segmento sonoro; phonological: frame de
  phonet). Se avisa de los audios con menos de 20 frames.

### `gita.caracteristicas` — limpieza, IDs y selección de características

Funciona con cualquier característica (audio, IMU, EEG...). Todas las funciones aceptan
un DataFrame, un dict `{dimension: DataFrame}` o la ruta a un `.pkl`, y devuelven la
misma estructura. Columnas de metadatos: `id` (obligatoria), `ruta` y `frame` (dinámicas).

```python
from gita.caracteristicas import cargar_pkl, limpiar_nan, obtener_ids, seleccionar_caracteristicas

c = cargar_pkl("caracteristicas/disvoice_estaticas.pkl")       # {"phonation": df, ...}

ids = obtener_ids(c, ruta_txt="ids_validos.txt")                # ids en todas las dimensiones
r = seleccionar_caracteristicas(c, "ids.csv", columna="id")     # o una lista de ids
c_filtrado = r["caracteristicas"]
```

| Función | Qué hace |
|---|---|
| `limpiar_nan(caracteristicas, estrategia="preguntar", eliminar_filas_vacias=True)` | 1) borra columnas todo-NaN; 2) borra filas todo-NaN (audios fallidos); 3) informa cuántas filas tienen NaN y aplica `media`, `mediana`, `moda`, `cero` o `eliminar` (estáticas: participantes; dinámicas: frames). `"preguntar"` lo pregunta en la consola. Un participante eliminado sale de todas las dimensiones |
| `obtener_ids(caracteristicas, ruta_txt=None)` | IDs presentes en todas las dimensiones (sin repetir, en orden); opcionalmente los guarda en un `.txt` legible con `leer_ids` |
| `seleccionar_caracteristicas(caracteristicas, ids, columna=None)` | Como `seleccionar_datos`, pero sobre características: filtra a los IDs de una lista o archivo `.txt`/`.csv`. Devuelve `caracteristicas` e `ids_no_encontrados` |
| `guardar_pkl(caracteristicas, ruta)` / `cargar_pkl(ruta)` | Guarda y carga un DataFrame o un dict de DataFrames |

> Imputar con estadísticos de todos los datos antes de dividir en entrenamiento/prueba
> filtra información de prueba al entrenamiento. Para los experimentos finales conviene
> imputar dentro de cada partición.

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
| `interseccion_ids(*listas, ruta_txt=None)` | IDs presentes en todas las listas (inner join), en el orden de la primera. Cada lista puede ser una lista de Python, un `.txt` o `(ruta_csv, columna)`. Ej.: `interseccion_ids(obtener_ids(c), "ids_gg.txt", ("demografia.csv", "id"))` |
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
