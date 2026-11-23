import numpy as np
import pandas as pd
import pytest

from gita.estadisticas import cargar_metadatos, prueba_categorica, prueba_continua


@pytest.fixture
def metadatos():
    rng = np.random.default_rng(1)
    n = 40
    return pd.DataFrame({
        "ID": ["S{:03d}".format(i) for i in range(2 * n)],
        "grupo": ["GG"] * n + ["GU"] * n,
        "edad": np.r_[rng.normal(40, 5, n), rng.normal(50, 5, n)],          # normal, distinta
        "moca": np.r_[rng.exponential(2, n), rng.exponential(2, n) + 3],    # sesgada, distinta
        "sexo": ["F"] * 25 + ["M"] * 15 + ["F"] * 20 + ["M"] * 20,           # 2x2, chi2 válido
        "raro": ["A"] * 38 + ["B"] * 2 + ["A"] * 30 + ["B"] * 10,           # esperadas < 5
    })


def test_continua_normal_usa_t(metadatos):
    r = prueba_continua(metadatos, "edad", "grupo", verbose=False)
    assert r["prueba"] in ("t de Student", "t de Welch")
    assert r["significativo"]
    assert r["n_por_grupo"] == {"GG": 40, "GU": 40}


def test_continua_no_normal_usa_mann_whitney(metadatos):
    r = prueba_continua(metadatos, "moca", "grupo", verbose=False)
    assert r["prueba"] == "U de Mann-Whitney"
    assert r["significativo"]


def test_continua_filtra_por_ids_y_reporta_faltantes(metadatos):
    ids = ["S000", "S001", "S002", "S040", "S041", "S042", "NOEXISTE"]
    r = prueba_continua(metadatos, "edad", "grupo", ids=ids, verbose=False)
    assert r["n_por_grupo"] == {"GG": 3, "GU": 3}
    assert r["ids_no_encontrados"] == ["NOEXISTE"]


def test_continua_tres_grupos(metadatos):
    df = metadatos.copy()
    df.loc[60:, "grupo"] = "Control"
    r = prueba_continua(df, "edad", "grupo", verbose=False)
    assert r["prueba"] in ("ANOVA de una vía", "ANOVA de Welch", "Kruskal-Wallis")


def test_continua_boxplot_guarda_figura(metadatos, tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    ruta = tmp_path / "edad.png"
    r = prueba_continua(metadatos, "edad", "grupo", boxplot=True, ruta_figura=ruta, verbose=False)
    assert ruta.is_file()
    assert r["figura"] is not None


def test_categorica_chi2_cuando_cumple_cochran(metadatos):
    r = prueba_categorica(metadatos, "sexo", "grupo", verbose=False)
    assert r["supuestos"]["cumple"]
    assert r["prueba"].startswith("Chi-cuadrado")
    assert r["tabla"].to_numpy().sum() == 80


def test_categorica_fisher_cuando_no_cumple(metadatos):
    df = metadatos.iloc[list(range(0, 10)) + list(range(40, 50))].copy()  # n = 20, esperadas < 5
    df["raro"] = ["A"] * 9 + ["B"] + ["A"] * 6 + ["B"] * 4
    r = prueba_categorica(df, "raro", "grupo", verbose=False)
    assert not r["supuestos"]["cumple"]
    assert r["prueba"] == "Exacta de Fisher"
    assert "odds ratio" in r["efecto"]


def test_categorica_rxc_fisher(metadatos):
    df = metadatos.iloc[list(range(0, 8)) + list(range(40, 48))].copy()
    df["nivel"] = ["bajo"] * 5 + ["medio"] * 2 + ["alto"] + ["bajo"] + ["medio"] * 2 + ["alto"] * 5
    r = prueba_categorica(df, "nivel", "grupo", n_simulaciones=2000, verbose=False)
    assert r["prueba"].startswith("Exacta de Fisher-Freeman-Halton")
    assert 0 <= r["p"] <= 1


def test_carga_desde_csv_conserva_ceros_en_ids(tmp_path):
    ruta = tmp_path / "meta.csv"
    ruta.write_text("ID;grupo;edad\n001;GG;40\n002;GU;50\n", encoding="utf-8")
    df = cargar_metadatos(ruta)
    assert df["ID"].tolist() == ["001", "002"]


def test_ids_repetidos_lanzan_error(metadatos):
    df = pd.concat([metadatos, metadatos.iloc[[0]]])
    with pytest.raises(ValueError, match="repetidos"):
        prueba_continua(df, "edad", "grupo", verbose=False)
