"""ANÁLISIS DE COLA IZQUIERDA — ¿avisan las señales negativas de las pérdidas severas?

Motivación: el retorno MEDIO tras una señal SELL es positivo (el mercado subió durante
todo el periodo). Eso no responde a la pregunta relevante para un inversionista, que no
es "¿cuánto gano de media?" sino "¿me avisa el sistema de las caídas fuertes?".

Se miden tres cosas, condicionadas a la señal emitida:
  1. Frecuencia de pérdidas severas, con umbral escalado al horizonte.
  2. Frecuencia de caer en el decil y el ventil peores de la distribución del horizonte.
  3. CVaR al 5 % (pérdida media en el 5 % de peores casos).

Se reporta además el `lift`: cuántas veces más probable es una pérdida severa tras SELL
que tras BUY. Un lift > 1 indica capacidad de aviso aunque la media sea positiva.

LIMITACIÓN QUE DEBE ACOMPAÑAR A ESTOS RESULTADOS: el universo (AAPL, MSFT, JPM) está
formado por tres empresas de gran éxito en el periodo. El sistema nunca se enfrentó a un
activo en declive estructural, así que estos resultados NO son extrapolables a ese caso.

Uso:  python -m src.reporting.tail_analysis
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import FIGURES, HORIZONS, METRICS, PREDICTIONS, TICKERS, ensure_dirs

# Umbral de "pérdida severa" escalado al horizonte (≈ −5 % a una semana)
SEVERE = {5: -0.05, 21: -0.10, 63: -0.15}
ORDER = ["BUY", "HOLD", "SELL"]
COL = {"BUY": "#1a7f37", "HOLD": "#8a8a8a", "SELL": "#c1121f"}

def tail_table(df: pd.DataFrame, by_ticker: bool = False) -> pd.DataFrame:
    keys = (["ticker", "horizon", "signal"] if by_ticker else ["horizon", "signal"])
    rec = []
    for k, g in df.groupby(keys):
        h = k[1] if by_ticker else k[0]
        r = g["realized_return_usd"].to_numpy()
        ref = df[df["horizon"] == h]["realized_return_usd"]
        q10, q05 = ref.quantile(0.10), ref.quantile(0.05)
        worst5 = np.sort(r)[:max(1, int(0.05 * len(r)))]
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row.update({
            "n": len(r),
            "p_severa": float((r < SEVERE[int(h)]).mean()),
            "p_decil_peor": float((r < q10).mean()),
            "p_ventil_peor": float((r < q05).mean()),
            "cvar_5": float(worst5.mean()),
            "peor_caso": float(r.min()),
            "media": float(r.mean()),
        })
        rec.append(row)
    return pd.DataFrame(rec)

def add_lift(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h, g in t.groupby("horizon"):
        g = g.set_index("signal")
        base = g.loc["BUY"]
        for s in ORDER:
            if s not in g.index:
                continue
            r = g.loc[s].to_dict()
            r["horizon"], r["signal"] = h, s
            r["lift_severa"] = (r["p_severa"] / base["p_severa"]
                                if base["p_severa"] > 0 else np.nan)
            r["lift_decil"] = (r["p_decil_peor"] / base["p_decil_peor"]
                               if base["p_decil_peor"] > 0 else np.nan)
            out.append(r)
    cols = ["horizon", "signal", "n", "media", "p_severa", "lift_severa",
            "p_decil_peor", "lift_decil", "p_ventil_peor", "cvar_5", "peor_caso"]
    return pd.DataFrame(out)[cols]

def figure(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (col, title) in zip(axes, [
            ("p_severa", "Probabilidad de pérdida severa"),
            ("p_decil_peor", "Probabilidad de caer en el decil peor"),
            ("cvar_5", "CVaR 5 % (pérdida media en el peor 5 %)")]):
        w, x = 0.26, np.arange(len(HORIZONS))
        for i, s in enumerate(ORDER):
            v = []
            for h in HORIZONS:
                sub = t[(t["horizon"] == h) & (t["signal"] == s)]
                v.append(float(sub[col].iloc[0]) if len(sub) else np.nan)
            ax.bar(x + (i - 1) * w, v, width=w - .02, color=COL[s], label=s)
        ax.set_xticks(x, [f"{h} s." for h in HORIZONS])
        ax.set_title(title, fontsize=8.5, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0)
    axes[1].axhline(0.10, color="k", ls="--", lw=.9)
    axes[1].text(0.02, .105, "10 % esperado si la señal no informara",
                 transform=axes[1].get_yaxis_transform(), fontsize=6.5, va="bottom")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Riesgo de cola por señal: las señales negativas sí concentran las caídas",
                 x=.01, ha="left", fontweight="bold", fontsize=9)
    fig.savefig(FIGURES / "fig13_cola_izquierda.png", bbox_inches="tight", dpi=150)
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    df = pd.read_parquet(PREDICTIONS / "signals.parquet")

    print("Base incondicional (contexto: el periodo fue alcista)")
    for h in HORIZONS:
        r = df[df["horizon"] == h]["realized_return_usd"]
        print(f"  h={h:<3} retornos negativos: {float((r < 0).mean()):.1%} | "
              f"pérdidas < {SEVERE[h]:.0%}: {float((r < SEVERE[h]).mean()):.1%}")

    t = add_lift(tail_table(df))
    t.to_csv(METRICS / "e5_cola_izquierda.csv", index=False)

    print("\n" + "=" * 92)
    print("RIESGO DE COLA CONDICIONADO A LA SEÑAL")
    print("=" * 92)
    show = t.copy()
    for c in ["media", "p_severa", "p_decil_peor", "p_ventil_peor", "cvar_5", "peor_caso"]:
        show[c] = show[c].map(lambda v: f"{v:+.1%}" if abs(v) < 10 else f"{v:.3f}")
    for c in ["lift_severa", "lift_decil"]:
        show[c] = show[c].map(lambda v: f"{v:.2f}×" if pd.notna(v) else "—")
    print(show.to_string(index=False))

    print("\n" + "=" * 92)
    print("LECTURA")
    print("=" * 92)
    for h in HORIZONS:
        g = t[t["horizon"] == h].set_index("signal")
        if not {"BUY", "SELL"} <= set(g.index):
            continue
        ls, ld = g.loc["SELL", "lift_severa"], g.loc["SELL", "lift_decil"]
        print(f"  h={h:<3} tras SELL una pérdida severa es {ls:.1f}× más probable que "
              f"tras BUY;\n       caer en el decil peor, {ld:.1f}×. "
              f"CVaR 5 %: BUY {g.loc['BUY','cvar_5']:+.1%} vs SELL {g.loc['SELL','cvar_5']:+.1%}")

    by_t = add_lift_by_ticker(df)
    by_t.to_csv(METRICS / "e5_cola_por_ticker.csv", index=False)
    print("\nDetalle por ticker guardado en e5_cola_por_ticker.csv")

    figure(t)
    print(f"\n✓ fig13_cola_izquierda.png")
    print("\n⚠ Limitación obligatoria en la memoria: el universo son tres empresas de")
    print("  gran éxito en el periodo. Estos resultados NO son extrapolables a activos")
    print("  en declive estructural, que el sistema nunca vio.")

def add_lift_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for t in TICKERS:
        sub = tail_table(df[df["ticker"] == t])
        sub = add_lift(sub)
        sub.insert(0, "ticker", t)
        out.append(sub)
    return pd.concat(out, ignore_index=True)

if __name__ == "__main__":
    main()
