"""Transacciones de clientes para CLV.

Fuente real recomendada: UCI "Online Retail II" (transacciones reales de un
e-commerce, 2009-2011, licencia CC BY 4.0):
  https://archive.ics.uci.edu/dataset/502/online+retail+ii
Descarga el .xlsx y guárdalo como data/online_retail_II.xlsx.
Si no existe, se genera una muestra sintética para probar el código.
"""
import numpy as np
import pandas as pd

import config

REAL = config.DATA_DIR / "online_retail_II.xlsx"
CACHE = config.DATA_DIR / "transacciones.csv"


def _desde_uci():
    hojas = pd.read_excel(REAL, sheet_name=None)
    df = pd.concat(hojas.values(), ignore_index=True)
    df = df.rename(columns={"Customer ID": "cliente_id", "InvoiceDate": "fecha", "Invoice": "factura"})
    df = df.dropna(subset=["cliente_id"])
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    df = df[~df["factura"].astype(str).str.startswith("C")]
    df["monto"] = df["Quantity"] * df["Price"]
    tx = df.groupby(["cliente_id", "factura"]).agg(fecha=("fecha", "min"), monto=("monto", "sum")).reset_index()
    return tx[["cliente_id", "fecha", "monto"]]


def _muestra(n_clientes=3000, seed=config.SEED):
    rng = np.random.default_rng(seed)
    inicio = pd.Timestamp.today().normalize() - pd.Timedelta(days=730)
    filas = []
    for cid in range(n_clientes):
        tasa = rng.gamma(1.2, 1 / 0.25)  # compras por año
        vida = rng.exponential(500)  # días activo
        ticket = rng.lognormal(np.log(config.TICKET_PROMEDIO), 0.4)
        alta = rng.uniform(0, 600)
        t = alta
        while t < min(alta + vida, 730):
            filas.append({"cliente_id": cid, "fecha": inicio + pd.Timedelta(days=float(t)),
                          "monto": round(ticket * rng.uniform(0.7, 1.3), 2)})
            t += rng.exponential(365 / max(tasa, 0.2))
    return pd.DataFrame(filas)


URL_UCI = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"


def descargar_uci():
    """Descarga automática del dataset UCI (funciona en Cloud Shell / Colab)."""
    import io
    import urllib.request
    import zipfile
    print("[UCI] Descargando Online Retail II (~44 MB)…")
    with urllib.request.urlopen(URL_UCI, timeout=120) as r:
        z = zipfile.ZipFile(io.BytesIO(r.read()))
    nombre = [n for n in z.namelist() if n.endswith(".xlsx")][0]
    REAL.write_bytes(z.read(nombre))


def cargar_transacciones():
    if not CACHE.exists() and not REAL.exists():
        try:
            descargar_uci()
        except Exception as e:
            print(f"[UCI] No se pudo descargar ({e}); se usará muestra sintética")
    if CACHE.exists():
        df = pd.read_csv(CACHE, parse_dates=["fecha"])
        fuente = (config.DATA_DIR / "tx_fuente.txt").read_text().strip()
        return df, fuente
    if REAL.exists():
        df, fuente = _desde_uci(), "UCI Online Retail II (transacciones reales)"
    else:
        df, fuente = _muestra(), "MUESTRA SINTÉTICA (reemplazar con UCI Online Retail II)"
    df.to_csv(CACHE, index=False)
    (config.DATA_DIR / "tx_fuente.txt").write_text(fuente)
    return df, fuente
