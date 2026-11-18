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
pip install git+https://github.com/Santiago6323/gita.git@v0.0.1
```

Actualizar a la última versión:

```bash
pip install --upgrade --force-reinstall git+https://github.com/Santiago6323/gita.git
```

En el `requirements.txt` de un proyecto:

```
gita-lab @ git+https://github.com/Santiago6323/gita.git@v0.0.1
```

> El paquete se instala como `gita-lab` (`pip uninstall gita-lab`) y se importa como `gita`.
> No uses `pip install gita`: ese nombre en PyPI pertenece a otra herramienta.

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
