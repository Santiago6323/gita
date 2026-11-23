"""
Pruebas estadísticas de comparación entre grupos a partir de una tabla de metadatos.

Dos funciones, una por tipo de variable:

- prueba_continua: variable numérica (edad, años de escolaridad, MoCA, ...).
    1. Normalidad por grupo (Shapiro-Wilk).
    2. Homogeneidad de varianzas (Levene), solo si todos los grupos son normales.
    3. Comparación:
         2 grupos:  t de Student / t de Welch   o   U de Mann-Whitney
        >2 grupos:  ANOVA de una vía / Welch    o   Kruskal-Wallis
- prueba_categorica: variable categórica (sexo, lateralidad, ...).
    1. Tabla de contingencia (variable x grupo) y frecuencias esperadas.
    2. Supuestos del chi-cuadrado (regla de Cochran).
    3. Chi-cuadrado de Pearson si se cumplen; si no, prueba exacta de Fisher
       (Fisher-Freeman-Halton en tablas mayores a 2x2).

Ambas reciben los metadatos como DataFrame o como ruta a un archivo
(.csv, .tsv, .txt, .xlsx, .xls) y, opcionalmente, una lista de IDs (o la ruta
a un archivo de IDs, que se lee con gita.datos.leer_ids) para analizar solo
esos sujetos.

Requiere pandas; matplotlib solo si se pide el diagrama de cajas.
"""

from pathlib import Path

import numpy as np
from scipy import stats

from gita.datos import leer_ids

# Colores del diagrama de cajas (paleta categórica validada para daltonismo)
_COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]


# --------------------------------------------------------------------------
# Carga y filtrado de metadatos (común a ambas pruebas)
# --------------------------------------------------------------------------

def cargar_metadatos(metadatos, col_id="ID"):
    """Devuelve una copia de los metadatos como DataFrame.

    metadatos: DataFrame o ruta a .csv/.tsv/.txt (separador detectado solo)
    o .xlsx/.xls. La columna col_id se convierte a str sin espacios para que
    coincida con los IDs leídos de archivos ("001" no es lo mismo que 1).
    """
    import pandas as pd

    if isinstance(metadatos, pd.DataFrame):
        df = metadatos.copy()
    else:
        ruta = Path(metadatos)
        if not ruta.is_file():
            raise FileNotFoundError("No existe el archivo de metadatos: {}".format(ruta))
        if ruta.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(ruta, dtype={col_id: str})
        else:
            df = pd.read_csv(ruta, sep=None, engine="python", dtype={col_id: str},
                             encoding="utf-8-sig")
        df.columns = [str(c).strip() for c in df.columns]

    if col_id not in df.columns:
        raise KeyError("No existe la columna de IDs '{}'. Columnas: {}".format(
            col_id, list(df.columns)))
    df[col_id] = df[col_id].astype(str).str.strip()
    return df


def _preparar(metadatos, variable, grupo, ids, col_id, verbose):
    """Carga, filtra por IDs, valida columnas y quita filas con NaN.

    Devuelve (df_limpio, info) con info = dict de conteos para el reporte.
    """
    df = cargar_metadatos(metadatos, col_id)
    for c in (variable, grupo):
        if c not in df.columns:
            raise KeyError("No existe la columna '{}'. Columnas: {}".format(c, list(df.columns)))

    duplicados = df[col_id][df[col_id].duplicated()].unique().tolist()
    if duplicados:
        raise ValueError("IDs repetidos en los metadatos (las observaciones deben ser "
                         "independientes): {}".format(duplicados[:10]))

    info = {"ids_no_encontrados": [], "n_excluidos_nan": 0}
    if ids is not None:
        if isinstance(ids, (str, Path)):
            ids = leer_ids(ids)
        ids = list(dict.fromkeys(str(i).strip() for i in ids))
        presentes = set(df[col_id])
        info["ids_no_encontrados"] = [i for i in ids if i not in presentes]
        df = df[df[col_id].isin(ids)]

    n_antes = len(df)
    df = df.dropna(subset=[variable, grupo])
    info["n_excluidos_nan"] = n_antes - len(df)

    if verbose:
        if info["ids_no_encontrados"]:
            print("IDs sin fila en los metadatos ({}): {}".format(
                len(info["ids_no_encontrados"]), ", ".join(info["ids_no_encontrados"][:10])))
        if info["n_excluidos_nan"]:
            print("{} sujetos excluidos por valores faltantes en '{}' o '{}'".format(
                info["n_excluidos_nan"], variable, grupo))

    if df[grupo].nunique() < 2:
        raise ValueError("Se necesitan al menos 2 grupos en '{}'; hay: {}".format(
            grupo, df[grupo].unique().tolist()))
    return df, info


def _formato_p(p):
    return "p < 0.001" if p < 0.001 else "p = {:.3f}".format(p)


# --------------------------------------------------------------------------
# Variable continua
# --------------------------------------------------------------------------

def prueba_continua(metadatos, variable, grupo, ids=None, col_id="ID", alfa=0.05,
                    boxplot=False, ruta_figura=None, verbose=True):
    """Compara una variable continua entre grupos, eligiendo la prueba según los supuestos.

    Parámetros
    ----------
    metadatos : DataFrame o ruta al archivo de metadatos.
    variable : columna numérica a comparar (p. ej. "edad").
    grupo : columna con la etiqueta de grupo (p. ej. "grupo" con GG / GU).
    ids : None (todos los sujetos), lista de IDs o ruta a un archivo de IDs.
    col_id : columna de los metadatos con el ID de cada sujeto.
    alfa : nivel de significancia de todas las pruebas.
    boxplot : True dibuja el diagrama de cajas y bigotes por grupo.
    ruta_figura : si se da, guarda ahí la figura (.png, .pdf, ...) en vez de mostrarla.
    verbose : imprime el reporte.

    Decisión
    --------
    - Normalidad: Shapiro-Wilk en cada grupo (H0: los datos son normales).
      Si algún grupo tiene p < alfa, o menos de 3 datos, se usa la vía no paramétrica.
    - Si todos son normales: Levene (H0: varianzas iguales) decide entre la
      versión clásica (varianzas iguales) y la de Welch.
    - 2 grupos: t de Student / t de Welch / U de Mann-Whitney.
      >2 grupos: ANOVA / ANOVA de Welch / Kruskal-Wallis.

    Devuelve un dict con: descriptivos (DataFrame), normalidad (dict por grupo),
    levene (dict o None), prueba, estadistico, p, significativo, efecto
    (nombre y valor), n_por_grupo, ids_no_encontrados, n_excluidos_nan y
    figura (Figure de matplotlib o None).
    """
    import pandas as pd

    df, info = _preparar(metadatos, variable, grupo, ids, col_id, verbose)
    df[variable] = pd.to_numeric(df[variable], errors="raise")

    nombres = sorted(df[grupo].unique(), key=str)
    muestras = [df.loc[df[grupo] == g, variable].to_numpy(dtype=float) for g in nombres]
    k = len(muestras)

    descriptivos = df.groupby(grupo)[variable].agg(
        n="count", media="mean", de="std", mediana="median",
        q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75),
        minimo="min", maximo="max",
    )

    # 1. Normalidad por grupo
    normalidad = {}
    for g, x in zip(nombres, muestras):
        if len(x) < 3:  # Shapiro-Wilk necesita al menos 3 datos
            normalidad[g] = {"W": np.nan, "p": np.nan, "normal": False}
        else:
            w, p = stats.shapiro(x)
            normalidad[g] = {"W": float(w), "p": float(p), "normal": bool(p >= alfa)}
    todos_normales = all(v["normal"] for v in normalidad.values())

    # 2. Homogeneidad de varianzas (solo relevante en la vía paramétrica)
    levene = None
    if todos_normales:
        est_l, p_l = stats.levene(*muestras, center="median")
        levene = {"estadistico": float(est_l), "p": float(p_l), "iguales": bool(p_l >= alfa)}

    # 3. Prueba de comparación y tamaño del efecto
    if k == 2:
        a, b = muestras
        if todos_normales:
            iguales = levene["iguales"]
            prueba = "t de Student" if iguales else "t de Welch"
            est, p = stats.ttest_ind(a, b, equal_var=iguales)
            # g de Hedges (d de Cohen con corrección por muestra pequeña)
            n1, n2 = len(a), len(b)
            de_comb = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
            d = (a.mean() - b.mean()) / de_comb
            efecto = ("g de Hedges", float(d * (1 - 3 / (4 * (n1 + n2) - 9))))
        else:
            prueba = "U de Mann-Whitney"
            est, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            # correlación biserial por rangos: r = 2U/(n1 n2) - 1, en [-1, 1]
            efecto = ("r biserial por rangos", float(2 * est / (len(a) * len(b)) - 1))
    else:
        n_total = sum(len(x) for x in muestras)
        if todos_normales and levene["iguales"]:
            prueba = "ANOVA de una vía"
            est, p = stats.f_oneway(*muestras)
        elif todos_normales:
            prueba = "ANOVA de Welch"
            est, p = _anova_welch(muestras)
        else:
            prueba = "Kruskal-Wallis"
            est, p = stats.kruskal(*muestras)
        if prueba == "Kruskal-Wallis":
            efecto = ("épsilon cuadrado", float(est * (n_total + 1) / (n_total ** 2 - 1)))
        else:
            media = np.concatenate(muestras).mean()
            ss_entre = sum(len(x) * (x.mean() - media) ** 2 for x in muestras)
            ss_total = sum(((x - media) ** 2).sum() for x in muestras)
            efecto = ("eta cuadrado", float(ss_entre / ss_total))

    resultado = {
        "variable": variable,
        "descriptivos": descriptivos,
        "normalidad": normalidad,
        "levene": levene,
        "prueba": prueba,
        "estadistico": float(est),
        "p": float(p),
        "significativo": bool(p < alfa),
        "efecto": efecto,
        "n_por_grupo": {g: len(x) for g, x in zip(nombres, muestras)},
        "ids_no_encontrados": info["ids_no_encontrados"],
        "n_excluidos_nan": info["n_excluidos_nan"],
        "figura": None,
    }

    if verbose:
        print("\n=== {} por {} ===".format(variable, grupo))
        print(descriptivos.round(3).to_string())
        print("\nNormalidad (Shapiro-Wilk, alfa = {}):".format(alfa))
        for g, v in normalidad.items():
            if np.isnan(v["p"]):
                print("  {}: menos de 3 datos -> se asume no normal".format(g))
            else:
                print("  {}: W = {:.3f}, {} -> {}".format(
                    g, v["W"], _formato_p(v["p"]), "normal" if v["normal"] else "NO normal"))
        if levene is not None:
            print("Levene: {} -> varianzas {}".format(
                _formato_p(levene["p"]), "iguales" if levene["iguales"] else "distintas"))
        print("\n{}: estadístico = {:.3f}, {} -> {}".format(
            prueba, resultado["estadistico"], _formato_p(p),
            "diferencia significativa" if resultado["significativo"] else "sin diferencia significativa"))
        print("Tamaño del efecto ({}): {:.3f}".format(*efecto))

    if boxplot:
        resultado["figura"] = _diagrama_cajas(nombres, muestras, variable, grupo,
                                              prueba, p, ruta_figura)
    return resultado


def _anova_welch(muestras):
    """ANOVA de Welch (varianzas distintas). Devuelve (F, p)."""
    k = len(muestras)
    n = np.array([len(x) for x in muestras], dtype=float)
    medias = np.array([x.mean() for x in muestras])
    var = np.array([x.var(ddof=1) for x in muestras])
    w = n / var
    media_pond = (w * medias).sum() / w.sum()
    a = (w * (medias - media_pond) ** 2).sum() / (k - 1)
    t = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * t
    f = a / b
    gl2 = (k ** 2 - 1) / (3 * t)
    return f, stats.f.sf(f, k - 1, gl2)


def _diagrama_cajas(nombres, muestras, variable, grupo, prueba, p, ruta_figura):
    """Cajas y bigotes por grupo con los datos individuales superpuestos."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.6 + 1.4 * len(nombres), 4.5))
    cajas = ax.boxplot(muestras, widths=0.5, patch_artist=True, showfliers=False,
                       medianprops={"color": "#1a1a19", "linewidth": 2})
    for i, caja in enumerate(cajas["boxes"]):
        color = _COLORES[i % len(_COLORES)]
        caja.set_facecolor(color)
        caja.set_alpha(0.35)
        caja.set_edgecolor(color)
    # datos individuales con dispersión horizontal para que no se solapen
    rng = np.random.default_rng(0)
    for i, x in enumerate(muestras, start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(x)), x, s=18, zorder=3,
                   color=_COLORES[(i - 1) % len(_COLORES)], edgecolor="white", linewidth=0.6)

    ax.set_xticks(range(1, len(nombres) + 1))
    ax.set_xticklabels(["{}\n(n = {})".format(g, len(x)) for g, x in zip(nombres, muestras)])
    ax.set_xlabel(grupo)
    ax.set_ylabel(variable)
    ax.set_title("{}: {}".format(prueba, _formato_p(p)), fontsize=10)
    ax.grid(axis="y", color="#e0e0e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    fig.tight_layout()

    if ruta_figura is not None:
        fig.savefig(ruta_figura, dpi=300)
        plt.close(fig)
    else:
        plt.show()
    return fig


# --------------------------------------------------------------------------
# Variable categórica
# --------------------------------------------------------------------------

def prueba_categorica(metadatos, variable, grupo, ids=None, col_id="ID", alfa=0.05,
                      n_simulaciones=100000, semilla=0, verbose=True):
    """Prueba de independencia entre una variable categórica y el grupo.

    Parámetros
    ----------
    metadatos, variable, grupo, ids, col_id, alfa, verbose : como en prueba_continua.
    n_simulaciones, semilla : solo se usan si la tabla es mayor que 2x2 y la
        versión de scipy no trae Fisher exacto para RxC; entonces el p-valor
        de Fisher-Freeman-Halton se estima por Monte Carlo.

    Supuestos del chi-cuadrado (si alguno falla se usa Fisher exacto)
    -------------------------------------------------------------------
    - Regla de Cochran: ninguna frecuencia esperada < 1 y como máximo el 20 %
      de las celdas con frecuencia esperada < 5.
    - En tablas 2x2 (criterio de Cochran para 2x2): n total >= 20 y, además,
      todas las esperadas >= 5. En 2x2 el chi-cuadrado usa corrección de Yates.
    - Observaciones independientes: cada sujeto aparece una sola vez (se
      verifica que no haya IDs repetidos).

    Devuelve un dict con: tabla (contingencia observada), esperadas,
    supuestos (dict con cada condición), prueba, estadistico, p, gl (None en
    Fisher), significativo, efecto (V de Cramér; en 2x2 también odds ratio),
    ids_no_encontrados y n_excluidos_nan.
    """
    import pandas as pd

    df, info = _preparar(metadatos, variable, grupo, ids, col_id, verbose)
    tabla = pd.crosstab(df[variable], df[grupo])
    if tabla.shape[0] < 2:
        raise ValueError("La variable '{}' tiene una sola categoría: {}".format(
            variable, tabla.index.tolist()))

    obs = tabla.to_numpy()
    n = int(obs.sum())
    esperadas_arr = stats.contingency.expected_freq(obs)
    esperadas = pd.DataFrame(esperadas_arr, index=tabla.index, columns=tabla.columns)
    es_2x2 = obs.shape == (2, 2)

    pct_menor_5 = float((esperadas_arr < 5).mean() * 100)
    supuestos = {
        "min_esperada": float(esperadas_arr.min()),
        "ninguna_esperada_menor_1": bool((esperadas_arr >= 1).all()),
        "pct_celdas_esperada_menor_5": pct_menor_5,
        "max_20pct_menor_5": bool(pct_menor_5 <= 20),
    }
    cumple = supuestos["ninguna_esperada_menor_1"] and supuestos["max_20pct_menor_5"]
    if es_2x2:
        supuestos["n_total_mayor_igual_20"] = bool(n >= 20)
        supuestos["todas_esperadas_mayor_igual_5"] = bool((esperadas_arr >= 5).all())
        cumple = cumple and supuestos["n_total_mayor_igual_20"] and supuestos["todas_esperadas_mayor_igual_5"]
    supuestos["cumple"] = bool(cumple)

    # V de Cramér (sin corrección de Yates, para que sea comparable entre tablas)
    chi2_sin_yates = stats.chi2_contingency(obs, correction=False)[0]
    v_cramer = float(np.sqrt(chi2_sin_yates / (n * (min(obs.shape) - 1))))
    efecto = {"V de Cramér": v_cramer}

    gl = None
    if cumple:
        est, p, gl, _ = stats.chi2_contingency(obs, correction=True)  # Yates solo afecta 2x2
        prueba = "Chi-cuadrado de Pearson" + (" (corrección de Yates)" if es_2x2 else "")
        gl = int(gl)
    elif es_2x2:
        prueba = "Exacta de Fisher"
        est, p = stats.fisher_exact(obs)
    else:
        prueba = "Exacta de Fisher-Freeman-Halton"
        est, p = _fisher_rxc(obs, n_simulaciones, semilla)
        if np.isnan(est):
            prueba += " (Monte Carlo, {} simulaciones)".format(n_simulaciones)

    if es_2x2:
        efecto["odds ratio"] = float(stats.contingency.odds_ratio(obs).statistic)

    resultado = {
        "variable": variable,
        "tabla": tabla,
        "esperadas": esperadas,
        "supuestos": supuestos,
        "prueba": prueba,
        "estadistico": float(est),
        "p": float(p),
        "gl": gl,
        "significativo": bool(p < alfa),
        "efecto": efecto,
        "ids_no_encontrados": info["ids_no_encontrados"],
        "n_excluidos_nan": info["n_excluidos_nan"],
    }

    if verbose:
        print("\n=== {} por {} ===".format(variable, grupo))
        print("Tabla de contingencia (observadas):")
        print(tabla.to_string())
        print("\nFrecuencias esperadas:")
        print(esperadas.round(2).to_string())
        print("\nSupuestos del chi-cuadrado:")
        print("  Esperada mínima: {:.2f} (debe ser >= 1) -> {}".format(
            supuestos["min_esperada"], "ok" if supuestos["ninguna_esperada_menor_1"] else "NO cumple"))
        print("  Celdas con esperada < 5: {:.0f} % (máx. 20 %) -> {}".format(
            pct_menor_5, "ok" if supuestos["max_20pct_menor_5"] else "NO cumple"))
        if es_2x2:
            print("  Tabla 2x2: n = {} (>= 20) y todas las esperadas >= 5 -> {}".format(
                n, "ok" if supuestos["n_total_mayor_igual_20"] and supuestos["todas_esperadas_mayor_igual_5"]
                else "NO cumple"))
        texto_gl = " (gl = {})".format(gl) if gl is not None else ""
        texto_est = "" if np.isnan(resultado["estadistico"]) else "estadístico = {:.3f}{}, ".format(
            resultado["estadistico"], texto_gl)
        print("\n{}: {}{} -> {}".format(
            prueba, texto_est, _formato_p(p),
            "asociación significativa" if resultado["significativo"] else "sin asociación significativa"))
        print("Tamaño del efecto: " + ", ".join("{} = {:.3f}".format(k, v) for k, v in efecto.items()))

    return resultado


def _fisher_rxc(obs, n_simulaciones, semilla):
    """Fisher exacto para tablas RxC. Devuelve (estadístico, p).

    scipy >= 1.15 lo calcula directamente con fisher_exact (el estadístico es
    la probabilidad de la tabla observada). En versiones anteriores se estima
    por Monte Carlo: se simulan tablas con los mismos totales de fila y
    columna y p = proporción de tablas tan o menos probables que la observada.
    En ese caso el estadístico se devuelve como NaN.
    """
    try:
        res = stats.fisher_exact(obs)
        return float(res.statistic), float(res.pvalue)
    except ValueError:  # scipy antiguo: solo 2x2
        pass
    dist = stats.random_table(obs.sum(axis=1), obs.sum(axis=0), seed=semilla)
    log_obs = dist.logpmf(obs)
    simuladas = dist.rvs(size=n_simulaciones)
    log_sim = dist.logpmf(simuladas)
    extremas = (log_sim <= log_obs + 1e-7).sum()
    return np.nan, float((extremas + 1) / (n_simulaciones + 1))
