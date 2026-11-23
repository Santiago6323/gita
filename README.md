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

### `gita.estadisticas` — comparación de grupos

Hace el ciclo completo de una prueba estadística entre grupos (p. ej. GG vs GU) a
partir de una tabla de metadatos: verifica los supuestos, elige la prueba, calcula el
tamaño del efecto e imprime un reporte. Los metadatos pueden ser un DataFrame o la ruta
a un `.csv`/`.tsv`/`.xlsx`, y se puede restringir el análisis a una lista de IDs (lista o
archivo, como en `gita.datos`). Dependencias: `pandas`; `matplotlib` para el diagrama
de cajas (`pip install "gita-lab[graficas]"`) y `openpyxl` para leer `.xlsx`
(`gita-lab[excel]`).

```python
from gita.estadisticas import prueba_continua, prueba_categorica

r = prueba_continua("metadatos.csv", "edad", "grupo", ids="ids.txt", boxplot=True)
print(r["prueba"], r["p"], r["efecto"])

r = prueba_categorica("metadatos.csv", "sexo", "grupo", ids="ids.txt")
print(r["tabla"])          # tabla de contingencia
print(r["supuestos"])      # regla de Cochran, esperada mínima, ...
```

Decisión de cada función:

- **Continua**: Shapiro-Wilk en cada grupo → si todos son normales, Levene decide entre
  t de Student (varianzas iguales) y t de Welch; si alguno no es normal (o tiene < 3
  datos), U de Mann-Whitney. Con más de 2 grupos: ANOVA / ANOVA de Welch / Kruskal-Wallis.
  Efecto: g de Hedges, r biserial por rangos, eta² o épsilon².
- **Categórica**: tabla de contingencia y frecuencias esperadas → chi-cuadrado (con
  corrección de Yates en 2x2) si se cumple la regla de Cochran (ninguna esperada < 1 y
  ≤ 20 % de celdas con esperada < 5; en 2x2 además n ≥ 20 y todas las esperadas ≥ 5);
  si no, Fisher exacta (Fisher-Freeman-Halton en tablas mayores, por Monte Carlo si la
  versión de scipy no la trae). Efecto: V de Cramér y, en 2x2, odds ratio.

En ambas se exige que no haya IDs repetidos (observaciones independientes), se excluyen
las filas con valores faltantes y se informa qué IDs de la lista no están en los metadatos.

| Función | Qué hace |
|---|---|
| `cargar_metadatos(metadatos, col_id="ID")` | DataFrame o ruta (`.csv`/`.tsv`/`.txt` con separador detectado solo, `.xlsx`/`.xls`) → DataFrame con los IDs como texto (conserva `001`) |
| `prueba_continua(metadatos, variable, grupo, ids=None, col_id="ID", alfa=0.05, boxplot=False, ruta_figura=None, verbose=True)` | Variable numérica por grupo; devuelve un dict con `descriptivos`, `normalidad`, `levene`, `prueba`, `estadistico`, `p`, `significativo`, `efecto`, `figura`, ... |
| `prueba_categorica(metadatos, variable, grupo, ids=None, col_id="ID", alfa=0.05, n_simulaciones=100000, semilla=0, verbose=True)` | Variable categórica por grupo; devuelve un dict con `tabla`, `esperadas`, `supuestos`, `prueba`, `estadistico`, `p`, `gl`, `significativo`, `efecto`, ... |

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
