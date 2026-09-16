"""SPIKE PLAN B — Alpaca News API (histórico desde 2015, cuenta gratuita).

Requiere un fichero `.env` en la raíz del proyecto con:

    ALPACA_KEY=PK...
    ALPACA_SECRET=...

(`.env` ya está en .gitignore: las claves NO se suben al repositorio.)

Uso:  python -m src.sentiment.spike_alpaca
"""
from __future__ import annotations
import os
import time
import pandas as pd
import requests
from src.config import ROOT, TICKERS

URL = "https://data.alpaca.markets/v1beta1/news"

def load_env() -> tuple[str, str]:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key, sec = os.environ.get("ALPACA_KEY", ""), os.environ.get("ALPACA_SECRET", "")
    if not key or not sec:
        raise SystemExit("Faltan ALPACA_KEY / ALPACA_SECRET en .env")
    return key, sec

def probe(key: str, sec: str, start: str, label: str) -> tuple[int, float]:
    params = {"symbols": ",".join(TICKERS), "start": f"{start}T00:00:00Z",
              "end": f"{start[:4]}-12-31T23:59:59Z", "limit": 50,
              "sort": "asc", "include_content": "false"}
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    t0 = time.time()
    try:
        r = requests.get(URL, params=params, headers=headers, timeout=30)
        el = time.time() - t0
        if r.status_code != 200:
            print(f"  {label}: HTTP {r.status_code} — {r.text[:120]}")
            return 0, el
        news = r.json().get("news", [])
        if not news:
            print(f"  {label}: 0 titulares ({el:.1f}s)")
            return 0, el
        ds = pd.to_datetime([n["created_at"] for n in news], utc=True, errors="coerce")
        print(f"  {label}: {len(news)} titulares | {ds.min():%Y-%m-%d} → {ds.max():%Y-%m-%d}"
              f" | {el:.1f}s")
        print(f"       [{news[0].get('symbols')}] {news[0].get('headline','')[:80]}")
        return len(news), el
    except Exception as e:
        print(f"  {label}: [error] {e}")
        return 0, time.time() - t0

def main() -> None:
    key, sec = load_env()
    print("--- Alpaca News API ---")
    res = [probe(key, sec, y, y[:4]) for y in
           ["2015-01-01", "2018-01-01", "2020-03-01", "2023-01-01", "2026-01-01"]]

    ok = [n for n, _ in res if n > 0]
    avg = sum(t for _, t in res) / len(res)
    print("\n--- VEREDICTO ---")
    if len(ok) >= 4:
        pages = 3 * 11 * 12 * 4   # tickers × años × meses × páginas estimadas
        print(f"  Alpaca sirve. Cobertura confirmada en {len(ok)}/5 sondas.")
        print(f"  Descarga completa 2015-2026: ~{pages} páginas × {avg:.1f}s "
              f"≈ {pages * avg / 60:.0f} min con caché incremental.")
        print("  Ventana del Experimento 2: 2015-2026 (~11 años).")
    elif ok:
        y = "el primer año con datos"
        print(f"  Cobertura parcial: acotar el Experimento 2 desde {y} y documentarlo.")
    else:
        print("  Sin acceso. Revisa que las claves sean de Paper Trading y estén activas.")
        print("  Si el problema persiste → PLAN C (ver conversación).")

if __name__ == "__main__":
    main()
