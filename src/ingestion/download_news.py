"""BRONZE (noticias) — Descarga incremental del histórico de Alpaca News API.

Guarda un parquet por año en data/bronze/news/ y puede reanudarse: si se corta,
vuelve a lanzarlo y continúa por el año que faltaba.

Campos conservados: id, created_at (UTC), headline, summary, source, author,
symbols (lista de tickers mencionados), url.

Uso:  python -m src.ingestion.download_news
"""
from __future__ import annotations
import json
import time
from datetime import datetime, timezone
import pandas as pd
import requests
from src.config import BRONZE, TICKERS, ensure_dirs
from src.sentiment.spike_alpaca import load_env

URL = "https://data.alpaca.markets/v1beta1/news"
START_YEAR, END_DATE = 2015, "2026-09-11"
PAGE_LIMIT = 50
PAUSE = 0.05          # cortesía con la API

def fetch_year(year: int, key: str, sec: str) -> pd.DataFrame:
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    start = f"{year}-01-01T00:00:00Z"
    end = min(f"{year}-12-31T23:59:59Z", f"{END_DATE}T23:59:59Z")
    if start[:10] > END_DATE:
        return pd.DataFrame()

    rows, token, pages = [], None, 0
    while True:
        params = {"symbols": ",".join(TICKERS), "start": start, "end": end,
                  "limit": PAGE_LIMIT, "sort": "asc", "include_content": "false"}
        if token:
            params["page_token"] = token
        r = requests.get(URL, params=params, headers=headers, timeout=45)
        if r.status_code == 429:                 # límite de ritmo: esperar y reintentar
            time.sleep(5)
            continue
        r.raise_for_status()
        payload = r.json()
        news = payload.get("news", [])
        for n in news:
            rows.append({
                "id": n.get("id"),
                "created_at": n.get("created_at"),
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "author": n.get("author"),
                "symbols": ",".join(n.get("symbols", [])),
                "url": n.get("url"),
            })
        pages += 1
        token = payload.get("next_page_token")
        if pages % 20 == 0:
            print(f"    {year}: {len(rows)} titulares...", end="\r")
        if not token or not news:
            break
        time.sleep(PAUSE)

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["created_at", "headline"]).drop_duplicates("id")
    return df.sort_values("created_at").reset_index(drop=True)

def main() -> None:
    ensure_dirs()
    out_dir = BRONZE / "news"
    out_dir.mkdir(parents=True, exist_ok=True)
    key, sec = load_env()

    total, t0, summary = 0, time.time(), {}
    for year in range(START_YEAR, int(END_DATE[:4]) + 1):
        path = out_dir / f"news_{year}.parquet"
        if path.exists():                         # reanudación
            df = pd.read_parquet(path)
            print(f"  {year}: {len(df):>6} titulares (ya descargado)")
            total += len(df)
            summary[year] = len(df)
            continue
        df = fetch_year(year, key, sec)
        if df.empty:
            print(f"  {year}: sin datos")
            continue
        df.to_parquet(path, index=False)
        total += len(df)
        summary[year] = len(df)
        print(f"  {year}: {len(df):>6} titulares  "
              f"{df['created_at'].min():%Y-%m-%d} → {df['created_at'].max():%Y-%m-%d}"
              f"  ({time.time() - t0:.0f}s)")

    (out_dir / "manifest.json").write_text(json.dumps({
        "source": "Alpaca News API (Benzinga)", "symbols": TICKERS,
        "start_year": START_YEAR, "end_date": END_DATE, "total_headlines": total,
        "by_year": summary,
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    print(f"\n{total} titulares en {out_dir}  ({time.time() - t0:.0f}s)")
    print("   Siguiente: FinBERT sobre los titulares (Fase 3).")

if __name__ == "__main__":
    main()
