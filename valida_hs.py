import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

boias = {"41004": [2023, 2024, 2025], "41040": [2019, 2020, 2021], "46080": [2023, 2024, 2025]}
locais = {"41004": "Edisto, Carolina do Sul",
          "41040": "Atlântico Norte, a leste das Pequenas Antilhas",
          "46080": "Portlock Bank, Golfo do Alasca"}

def carregar(id_boia):
    dfs = []
    for ano in boias[id_boia]:
        caminho = os.path.join("dados", f"{id_boia}_{ano}", f"series_temporais_{id_boia}_{ano}.csv")
        df = pd.read_csv(caminho, parse_dates=["data_registro"])
        if "height" not in df.columns:
            raise SystemExit(f"{caminho} sem coluna 'height' (rode wind.py e analise.py).")
        dfs.append(df)
    df = pd.concat(dfs).drop_duplicates(subset="data_registro")
    return df[["data_registro", "Hs", "height"]].dropna()

virg = lambda x, casas: f"{x:.{casas}f}".replace(".", ",")

for id_boia in boias:
    df = carregar(id_boia)
    x, y = df["Hs"].values, df["height"].values
    n = len(df)
    erro_medio = (x - y).mean()
    sinal = "+" if erro_medio >= 0 else "−"
    periodo = f"{boias[id_boia][0]} a {boias[id_boia][-1]}"

    lim = max(x.max(), y.max()) * 1.05
    fig, ax = plt.subplots(figsize=(7, 6.5))
    hb = ax.hexbin(x, y, gridsize=70, extent=(0, lim, 0, lim), mincnt=1,
                   cmap="viridis", norm=LogNorm(), linewidths=0.1)
    ax.plot([0, lim], [0, lim], color="red", ls="--", lw=1.2, label="$y=x$")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("$H_s$ calculado (m)")
    ax.set_ylabel("WVHT publicado pelo NDBC (m)")
    ax.set_title(f"Boia {id_boia} ({locais[id_boia]})\nPeríodo {periodo}")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    cb = fig.colorbar(hb, ax=ax, shrink=0.8)
    cb.set_label("Número de registros por célula (escala log)")
    ax.text(0.03, 0.97,
            f"N = {n} registros pareados\nerro médio = {sinal}{virg(abs(erro_medio), 2)} m\n(média de $H_s$ − WVHT)",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.9))
    plt.savefig(os.path.join("dados", f"validacao_hs_{id_boia}.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(id_boia, n, virg(erro_medio, 3))