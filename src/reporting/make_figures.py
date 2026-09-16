"""FIGURAS — Genera los gráficos de la memoria a partir de los artefactos congelados.

No recalcula nada: lee outputs/metrics, outputs/predictions y data/gold. Si los
resultados están congelados, las figuras son deterministas.

Uso:  python -m src.reporting.make_figures
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.config import FIGURES, GOLD, METRICS, PREDICTIONS, SILVER, TICKERS, ensure_dirs

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.autolayout": True,
})
C = {"AAPL": "#0b6e4f", "MSFT": "#1d4e89", "JPM": "#a4303f", "SPY": "#6b6b6b"}
DIV = "RdYlGn"

def _save(fig, name: str) -> None:
    p = FIGURES / f"{name}.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {p.name}")

def _heat(ax, data: pd.DataFrame, vmax: float, fmt: str = "{:+.3f}", cmap: str = DIV):
    ax.grid(False)
    im = ax.imshow(data.to_numpy(dtype=float), cmap=cmap, vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_xticks(range(len(data.columns)), data.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(data.index)),
                  [" · ".join(map(str, i)) if isinstance(i, tuple) else str(i)
                   for i in data.index])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.to_numpy(dtype=float)[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="black")
    return im


def fig01_panel() -> None:
    """Contexto: precios normalizados, VIX y USD/MXN."""
    prices = pd.read_parquet(SILVER / "prices.parquet")
    ctx = pd.read_parquet(SILVER / "market_context.parquet")
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 1]})
    for t in TICKERS + ["SPY"]:
        s = prices[prices["ticker"] == t].sort_values("date")
        axes[0].plot(s["date"], s["adj_close"] / s["adj_close"].iloc[0],
                     label=t, color=C[t], lw=1.1)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Precio normalizado (log)")
    axes[0].legend(ncol=4, frameon=False, loc="upper left")
    axes[0].set_title("Panel de estudio · 2006-2026", loc="left", fontweight="bold")

    axes[1].fill_between(ctx["date"], ctx["vix_close"], color="#7a5195", alpha=.75, lw=0)
    axes[1].set_ylabel("VIX")
    axes[2].plot(ctx["date"], ctx["usdmxn"], color="#bc5090", lw=1.1)
    axes[2].set_ylabel("USD/MXN")
    axes[2].set_xlabel("Fecha")
    _save(fig, "fig01_panel_contexto")

def fig02_folds() -> None:
    """Esquema del walk-forward con purga."""
    pred = pd.read_parquet(PREDICTIONS / "e1_walkforward.parquet")
    g = pred[(pred["ticker"] == TICKERS[0]) & (pred["horizon"] == 21)]
    folds = (g.groupby("fold")["date"].agg(["min", "max"]).sort_index())
    fig, ax = plt.subplots(figsize=(9, 4))
    start = folds["min"].min() - pd.Timedelta(days=1825)
    for k, r in folds.iterrows():
        ax.barh(k, (r["min"] - start).days, left=start, height=.6,
                color="#c9d6df", label="Entrenamiento" if k == 0 else None)
        ax.barh(k, 21 * 1.45, left=r["min"] - pd.Timedelta(days=int(21 * 1.45)),
                height=.6, color="#f0a202", label="Purga (h sesiones)" if k == 0 else None)
        ax.barh(k, (r["max"] - r["min"]).days, left=r["min"], height=.6,
                color="#1d4e89", label="Prueba" if k == 0 else None)
    ax.set_yticks(folds.index, [f"Pliegue {k}" for k in folds.index], fontsize=7)
    ax.invert_yaxis()
    ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.02))
    ax.set_title("Validación walk-forward con purga del horizonte (h = 21)",
                 loc="left", fontweight="bold", pad=28)
    _save(fig, "fig02_walkforward")

def fig03_ic_modelos() -> None:
    """IC de Spearman medio por modelo y horizonte."""
    pm = pd.read_csv(METRICS / "e1_predictive.csv")
    pm = pm[~pm["model"].isin(["zero", "hist_mean"])]
    piv = pm.pivot_table(index="model", columns="horizon", values="spearman_ic")
    piv = piv.sort_values(piv.columns[-1])
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = np.arange(len(piv))
    for i, h in enumerate(piv.columns):
        ax.bar(x + (i - 1) * .27, piv[h], width=.26, label=f"{h} sesiones")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x, piv.index, rotation=20, ha="right")
    ax.set_ylabel("IC de Spearman")
    ax.legend(frameon=False, ncol=3)
    ax.set_title("Capacidad predictiva por modelo: la señal es lineal y crece con el horizonte",
                 loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig03_ic_por_modelo")

def fig04_heterogeneidad() -> None:
    """Hallazgo central: ningún modelo domina."""
    pm = pd.read_csv(METRICS / "e1_predictive.csv")
    pm = pm[~pm["model"].isin(["zero", "hist_mean"])]
    piv = pm.pivot_table(index=["ticker", "horizon"], columns="model",
                         values="spearman_ic")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = _heat(ax, piv, vmax=float(np.nanmax(np.abs(piv.to_numpy()))))
    fig.colorbar(im, ax=ax, shrink=.8, label="IC de Spearman")
    ax.set_title("Ningún modelo domina en todos los activos y horizontes",
                 loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig04_heterogeneidad")

def fig05_dir_acc() -> None:
    """Acierto direccional frente a su tasa base."""
    pm = pd.read_csv(METRICS / "e1_predictive.csv")
    pm = pm[pm["dir_acc"].notna()]
    agg = pm.groupby(["horizon", "model"])[["dir_acc", "base_rate"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, h in zip(axes, sorted(agg["horizon"].unique())):
        s = agg[agg["horizon"] == h].sort_values("dir_acc")
        ax.barh(s["model"], s["dir_acc"], color="#1d4e89", height=.6)
        ax.axvline(s["base_rate"].iloc[0], color="#c1121f", lw=1.6,
                   label="Tasa base (% de retornos positivos)")
        ax.set_xlim(0.45, 0.78)
        ax.set_title(f"{h} sesiones", fontsize=9)
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("El acierto direccional solo se interpreta frente a su tasa base",
                 x=.01, ha="left", fontweight="bold", fontsize=9)
    _save(fig, "fig05_dir_acc_vs_base")

def fig06_sentimiento() -> None:
    """E2: efecto del sentimiento por ticker."""
    try:
        cmp = pd.read_csv(METRICS / "e2_comparison.csv")
    except FileNotFoundError:
        return
    piv = cmp.pivot_table(index=["ticker", "horizon"], columns="model", values="delta_ic")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2),
                             gridspec_kw={"width_ratios": [3, 1.4]})
    im = _heat(axes[0], piv, vmax=float(np.nanmax(np.abs(piv.to_numpy()))))
    fig.colorbar(im, ax=axes[0], shrink=.8, label="Δ IC (sentimiento − mercado)")
    axes[0].set_title("Efecto del sentimiento, modelo a modelo", loc="left",
                      fontweight="bold", fontsize=9)

    win = cmp.groupby("ticker").apply(lambda d: float((d["delta_ic"] > 0).mean()),
                                      include_groups=False)
    axes[1].bar(win.index, win.to_numpy(), color=[C[t] for t in win.index], width=.55)
    axes[1].axhline(.5, color="k", ls="--", lw=1, label="Azar")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Casos en que el sentimiento mejora")
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_title("Depende del activo", loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig06_efecto_sentimiento")

def fig07_ensemble() -> None:
    """E3: ensembles frente a media simple y modelos individuales."""
    pm = pd.read_csv(METRICS / "e3_predictive.csv")
    pm = pm[pm["source"] == "e1"]
    agg = pm.groupby("model")["spearman_ic"].mean().sort_values()
    colors = ["#f0a202" if m.startswith("ens_") else "#c9d6df" for m in agg.index]
    colors = ["#c1121f" if m == "ens_equal_expanding" else c
              for m, c in zip(agg.index, colors)]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(agg.index, agg.to_numpy(), color=colors, height=.65)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("IC de Spearman medio")
    ax.set_title("Los esquemas adaptativos (naranja) superan a la media simple (rojo)",
                 loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig07_ensemble")

def fig08_peso_sentimiento() -> None:
    """El ensemble decide por sí solo dónde usar sentimiento."""
    try:
        w = pd.read_csv(METRICS / "e3_weights.csv")
    except FileNotFoundError:
        return
    sel = w[(w["source"] == "e2") & (w["scheme"] == "softmax_ic")
            & (w["window"] == "expanding") & (w["fold"] > 0)].copy()
    if sel.empty:
        return
    sel["sent"] = sel["model"].str.endswith("@sent")
    share = (sel.groupby(["ticker", "horizon"])
             .apply(lambda d: d.loc[d["sent"], "weight"].sum() / d["weight"].sum(),
                    include_groups=False).unstack(1))
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    ax.grid(False)
    im = ax.imshow(share.to_numpy() - .5, cmap=DIV, vmin=-.2, vmax=.2, aspect="auto")
    ax.set_xticks(range(len(share.columns)), [f"{c} sesiones" for c in share.columns])
    ax.set_yticks(range(len(share.index)), share.index)
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            ax.text(j, i, f"{share.to_numpy()[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=.85, label="Peso al sentimiento − 50 %")
    ax.set_title("El ensemble elige el sentimiento sin ver el futuro",
                 loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig08_peso_sentimiento")

def fig09_calidad_senal() -> None:
    """BUY / HOLD / SELL frente al resultado realizado."""
    sg = pd.read_parquet(PREDICTIONS / "signals.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    order = ["SELL", "HOLD", "BUY"]
    col = {"BUY": "#1a7f37", "HOLD": "#8a8a8a", "SELL": "#c1121f"}
    for ax, (metric, label) in zip(axes, [("mean", "Retorno medio realizado"),
                                          ("acc", "Proporción de resultados positivos")]):
        for i, h in enumerate(sorted(sg["horizon"].unique())):
            s = sg[sg["horizon"] == h]
            vals = []
            for sig in order:
                x = s[s["signal"] == sig]["realized_return_usd"]
                vals.append(x.mean() if metric == "mean" else float((x > 0).mean()))
            ax.plot(order, vals, "o-", label=f"{h} sesiones", lw=1.4, ms=5)
        ax.set_ylabel(label)
        for sig in order:
            ax.axvline(sig, color=col[sig], alpha=.12, lw=18)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle("La señal ordena correctamente los resultados (BUY > HOLD > SELL)",
                 x=.01, ha="left", fontweight="bold", fontsize=9)
    _save(fig, "fig09_calidad_senal")

def fig10_equity() -> None:
    """Curvas de capital: señales frente a buy & hold y SPY."""
    sg = pd.read_parquet(PREDICTIONS / "signals.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    for ax, t in zip(axes, TICKERS):
        s = sg[(sg["ticker"] == t) & (sg["horizon"] == 21)].sort_values("date").iloc[::21]
        if s.empty:
            continue
        pos = np.where(s["signal"] == "BUY", 1., np.where(s["signal"] == "SELL", -1., 0.))
        r = s["realized_return_usd"].to_numpy()
        ax.plot(s["date"], np.cumprod(1 + pos * r), label="Señales", color="#1d4e89", lw=1.4)
        ax.plot(s["date"], np.cumprod(1 + np.maximum(pos, 0) * r), label="Solo largos",
                color="#f0a202", lw=1.2)
        ax.plot(s["date"], np.cumprod(1 + r), label="Buy & hold", color="#6b6b6b",
                lw=1.2, ls="--")
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9, color=C[t], fontweight="bold")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("Capital acumulado (log)")
    fig.suptitle("Ninguna estrategia basada en las señales supera a buy & hold (h = 21)",
                 x=.01, ha="left", fontweight="bold", fontsize=9)
    _save(fig, "fig10_equity")

def fig11_divisa() -> None:
    """Efecto de la divisa para el inversionista mexicano."""
    sg = pd.read_parquet(PREDICTIONS / "signals.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    hs = sorted(sg["horizon"].unique())
    flips = [float((np.sign(sg.query("horizon == @h")["realized_return_usd"])
                    != np.sign(sg.query("horizon == @h")["realized_return_mxn"])).mean())
             for h in hs]
    axes[0].bar([f"{h} s." for h in hs], flips, color="#bc5090", width=.5)
    axes[0].set_ylabel("Proporción que cambia de signo")
    axes[0].set_title("Uno de cada siete resultados se invierte en pesos",
                      loc="left", fontweight="bold", fontsize=9)

    s = sg[sg["horizon"] == 21]
    axes[1].scatter(s["realized_return_usd"], s["realized_return_mxn"], s=3, alpha=.15,
                    color="#1d4e89", edgecolors="none")
    lim = float(np.nanpercentile(np.abs(s["realized_return_usd"]), 99.5))
    axes[1].plot([-lim, lim], [-lim, lim], color="k", lw=.9, ls="--")
    axes[1].axhline(0, color="#c1121f", lw=.7); axes[1].axvline(0, color="#c1121f", lw=.7)
    axes[1].set_xlim(-lim, lim); axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("Retorno realizado USD"); axes[1].set_ylabel("Retorno realizado MXN")
    axes[1].set_title("Descomposición activo × divisa (h = 21)", loc="left",
                      fontweight="bold", fontsize=9)
    _save(fig, "fig11_divisa")

def fig12_sentimiento_serie() -> None:
    """Serie de sentimiento y cobertura."""
    try:
        g = pd.read_parquet(GOLD / "market_sentiment.parquet")
    except FileNotFoundError:
        return
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.4), sharex=True)
    for t in TICKERS:
        s = g[g["ticker"] == t].sort_values("date")
        axes[0].plot(s["date"], s["sentiment_mean_7d"].rolling(21).mean(),
                     label=t, color=C[t], lw=1.1)
        axes[1].plot(s["date"], s["news_volume_7d"].rolling(21).mean(),
                     color=C[t], lw=1.1)
    axes[0].axhline(0, color="k", lw=.7)
    axes[0].set_ylabel("Polaridad FinBERT")
    axes[0].legend(frameon=False, ncol=3)
    axes[1].set_ylabel("Titulares / 7 sesiones")
    axes[1].set_xlabel("Fecha")
    axes[0].set_title("Sentimiento y cobertura informativa (media móvil 21 sesiones)",
                      loc="left", fontweight="bold", fontsize=9)
    _save(fig, "fig12_sentimiento")

def main() -> None:
    ensure_dirs()
    print(f"Generando figuras en {FIGURES}\n")
    for fn in (fig01_panel, fig02_folds, fig03_ic_modelos, fig04_heterogeneidad,
               fig05_dir_acc, fig06_sentimiento, fig07_ensemble, fig08_peso_sentimiento,
               fig09_calidad_senal, fig10_equity, fig11_divisa, fig12_sentimiento_serie):
        try:
            fn()
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {type(e).__name__}: {e}")
    n = len(list(FIGURES.glob("*.png")))
    print(f"\n {n} figuras listas para la memoria.")

if __name__ == "__main__":
    main()
