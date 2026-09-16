"""FinBERT — Clasificación de titulares en positive / neutral / negative.

Modelo: ProsusAI/finbert (BERT afinado sobre textos financieros).
Se puntúa cada titular ÚNICO una sola vez (no por ticker) y se guardan las tres
probabilidades. Guarda checkpoints: si se corta, se relanza y continúa.

Salida: silver/finbert_scores.parquet (id, p_negative, p_neutral, p_positive, score)
  `score` = p_positive - p_negative  ∈ [-1, 1]  (polaridad continua)

Uso:  python -m src.sentiment.finbert_score
"""
from __future__ import annotations
import time
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.config import SILVER, ensure_dirs

MODEL = "ProsusAI/finbert"
BATCH = 64
MAX_LEN = 64            # los titulares son cortos: recortar acelera mucho
CKPT_EVERY = 100        # lotes entre checkpoints
OUT = SILVER / "finbert_scores.parquet"

def main() -> None:
    ensure_dirs()
    news = pd.read_parquet(SILVER / "news.parquet")
    todo = news.drop_duplicates("id")[["id", "headline"]].reset_index(drop=True)

    done = pd.DataFrame()
    if OUT.exists():
        done = pd.read_parquet(OUT)
        todo = todo[~todo["id"].isin(done["id"])].reset_index(drop=True)
        print(f"Reanudando: {len(done)} ya puntuados, faltan {len(todo)}")

    if todo.empty:
        print("Todo puntuado.")
        return

    torch.set_num_threads(max(1, torch.get_num_threads()))
    print(f"Cargando {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

    # El orden de las etiquetas lo dicta el modelo, no se asume
    labels = [model.config.id2label[i].lower() for i in range(model.config.num_labels)]
    print(f"Etiquetas: {labels} | {len(todo)} titulares | lote {BATCH}")

    rows, t0 = [], time.time()
    n_batches = (len(todo) + BATCH - 1) // BATCH
    with torch.no_grad():
        for b in range(n_batches):
            chunk = todo.iloc[b * BATCH:(b + 1) * BATCH]
            enc = tok(chunk["headline"].tolist(), padding=True, truncation=True,
                      max_length=MAX_LEN, return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            part = pd.DataFrame(probs, columns=[f"p_{l}" for l in labels])
            part.insert(0, "id", chunk["id"].to_numpy())
            rows.append(part)

            if (b + 1) % CKPT_EVERY == 0 or b == n_batches - 1:
                cur = pd.concat([done] + rows, ignore_index=True)
                cur.to_parquet(OUT, index=False)
                el = time.time() - t0
                seen = min((b + 1) * BATCH, len(todo))
                eta = el / max(seen, 1) * (len(todo) - seen)
                print(f"  {seen}/{len(todo)}  {seen/el:.0f} titulares/s  "
                      f"ETA {eta/60:.1f} min", end="\r")

    out = pd.concat([done] + rows, ignore_index=True).drop_duplicates("id")
    out["score"] = out["p_positive"] - out["p_negative"]
    out.to_parquet(OUT, index=False)

    print(f"\n\n {len(out)} titulares puntuados en {(time.time() - t0)/60:.1f} min")
    dist = out[[c for c in out.columns if c.startswith("p_")]].idxmax(axis=1)
    print("Distribución:", dist.str.replace("p_", "").value_counts().to_dict())
    print(f"Polaridad media: {out['score'].mean():+.4f} "
          f"(desv. {out['score'].std():.4f})")

if __name__ == "__main__":
    main()
