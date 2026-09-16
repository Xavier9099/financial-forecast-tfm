"""SPIKE DE VIABILIDAD — ¿hay histórico de noticias utilizable para el Experimento 2?

No descarga el corpus. Sólo responde tres preguntas antes de comprometer el lunes:
  1. ¿Qué cobertura da yfinance? (esperado: sólo reciente)
  2. ¿GDELT 2.0 devuelve artículos de hace años con timestamp?
  3. ¿Cuánto tarda una petición? -> estimación del tiempo total de descarga.

Uso:  python -m src.sentiment.spike_news
"""
from __future__ import annotations
import time
from datetime import datetime
import pandas as pd
import requests
from src.config import TICKERS

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERIES = {"AAPL": '"Apple Inc"', "MSFT": '"Microsoft"', "JPM": '"JPMorgan"'}
UA = {"User-Agent": "TFM-UCM-academic-research/1.0"}

def probe_yfinance() -> None:
    print("\n--- 1. yfinance (Ticker.news) ---")
    try:
        import yfinance as yf
        for t in TICKERS:
            items = yf.Ticker(t).news or []
            dates = []
            for it in items:
                c = it.get("content", it)
                raw = c.get("pubDate") or c.get("providerPublishTime")
                if isinstance(raw, (int, float)):
                    dates.append(datetime.utcfromtimestamp(raw))
                elif isinstance(raw, str):
                    dates.append(pd.to_datetime(raw, errors="coerce", utc=True))
            dates = [d for d in dates if pd.notna(d)]
            rng = f"{min(dates):%Y-%m-%d} → {max(dates):%Y-%m-%d}" if dates else "sin fechas"
            print(f"  {t}: {len(items)} noticias | {rng}")
    except Exception as e:
        print(f"  [error] {e}")
    print("  Veredicto esperado: sólo días recientes -> NO sirve como histórico.")

def probe_gdelt(week_start: str, label: str) -> tuple[int, float]:
    start = pd.Timestamp(week_start)
    params = {
        "query": f'{QUERIES["AAPL"]} sourcelang:english',
        "mode": "artlist", "format": "json", "maxrecords": 250,
        "startdatetime": start.strftime("%Y%m%d000000"),
        "enddatetime": (start + pd.Timedelta(days=7)).strftime("%Y%m%d000000"),
        "sort": "hybridrel",
    }
    t0 = time.time()
    try:
        r = requests.get(GDELT, params=params, headers=UA, timeout=45)
        elapsed = time.time() - t0
        if r.status_code != 200:
            print(f"  {label}: HTTP {r.status_code} ({elapsed:.1f}s)")
            return 0, elapsed
        arts = r.json().get("articles", [])
        if arts:
            ds = pd.to_datetime([a.get("seendate") for a in arts], errors="coerce", utc=True)
            ds = ds.dropna()
            rng = f"{ds.min():%Y-%m-%d} → {ds.max():%Y-%m-%d}" if len(ds) else "sin fechas"
            print(f"  {label}: {len(arts):>3} artículos | {rng} | {elapsed:.1f}s")
            print(f"       ejemplo: {arts[0].get('title', '')[:90]}")
        else:
            print(f"  {label}: 0 artículos ({elapsed:.1f}s)")
        return len(arts), elapsed
    except Exception as e:
        print(f"  {label}: [error] {e}")
        return 0, time.time() - t0

def main() -> None:
    probe_yfinance()

    print("\n--- 2. GDELT 2.0 Doc API (sin API key) ---")
    probes = [("2017-01-09", "2017 (inicio de cobertura v2)"),
              ("2020-03-16", "2020 (crash COVID)"),
              ("2024-06-10", "2024"),
              ("2026-08-31", "2026 (reciente)")]
    results = [probe_gdelt(d, lbl) for d, lbl in probes]

    ok = [n for n, _ in results if n > 0]
    avg = sum(t for _, t in results) / len(results)
    weeks = 3 * int((pd.Timestamp("2026-09-10") - pd.Timestamp("2017-01-01")).days / 7)

    print("\n--- VEREDICTO ---")
    if len(ok) >= 3:
        print(f"  GDELT sirve como histórico. ~{sum(ok)/len(ok):.0f} art./semana/ticker.")
        print(f"  Descarga completa 2017-2026: ~{weeks} peticiones × {avg:.1f}s "
              f"≈ {weeks * avg / 60:.0f} min. Ejecutar el lunes con caché incremental.")
        print("  Ventana del Experimento 2: 2017-2026 (~9 años). Suficiente.")
    elif ok:
        print("  Cobertura parcial. Acotar el Experimento 2 a los años con datos")
        print("   y documentarlo como limitación en la memoria.")
    else:
        print("  GDELT no responde. PLAN B: crear cuenta gratuita en Alpaca")
        print("  (alpaca.markets) -> News API con histórico desde 2015.")
        print("  PLAN C: Experimento 2 sobre ventana corta con yfinance + declarar")
        print("  la limitación. El TFM sigue siendo válido sin E2 completo.")

if __name__ == "__main__":
    main()
