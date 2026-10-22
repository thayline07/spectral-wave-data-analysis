import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Boias escolhidas e anos usados em cada uma
boias = {
    "41004": [2023, 2024, 2025],
    "41040": [2019, 2020, 2021],
    "46080": [2023, 2024, 2025],
}

# Nome da localidade 
locais = {
    "41004": "Edisto, Carolina do Sul",
    "41040": "Atlântico Norte, a leste das Pequenas Antilhas",
    "46080": "Portlock Bank, Golfo do Alasca",
}

id_boia = "46080"   # trocar para gerar as figuras de cada boia

anos = boias[id_boia]
local = locais[id_boia]
periodo = f"{anos[0]}\u2013{anos[-1]}"

# Lê e junta todos os anos da boia
partes = []
for year in anos:
    caminho_csv = os.path.join(
        "dados", f"{id_boia}_{year}", f"series_temporais_{id_boia}_{year}.csv"
    )
    if os.path.exists(caminho_csv):
        partes.append(pd.read_csv(caminho_csv, parse_dates=["data_registro"]))
    else:
        print(f"Aviso: {caminho_csv} não encontrado, ano {year} ficou de fora.")

if not partes:
    print("Nenhum arquivo encontrado.")
    exit(1)

df_series = (
    pd.concat(partes)
    .drop_duplicates(subset="data_registro")
    .set_index("data_registro")
    .sort_index()
)
print(f"Boia {id_boia}: {len(df_series)} registros ({periodo})")

# Histogramas

# Altura significativa
altura_maxima = df_series["Hs"].max()
mediana_hs = df_series["Hs"].median()
percentil_99 = df_series["Hs"].quantile(0.99)
print(mediana_hs, altura_maxima, percentil_99)


def checar_vizinhos(df, valor_max, n_vizinhos=5):
    hs = df["Hs"]
    pos = (hs - valor_max).abs().to_numpy().argmin()
    inicio = max(pos - n_vizinhos, 0)
    fim = pos + n_vizinhos + 1
    return df.iloc[inicio:fim][["Hs"]]


print(f"\nVizinhança do Hs máximo (boia {id_boia}, {periodo}):")
print(checar_vizinhos(df_series, altura_maxima))

plt.figure(figsize=(7, 5))
plt.axvline(x=mediana_hs, color="red", linestyle="--", label=f"Mediana: {mediana_hs:.2f} m")
plt.axvline(x=altura_maxima, color="green", linestyle="--", label=f"Hs Máximo: {altura_maxima:.2f} m")
plt.axvline(x=percentil_99, color="orange", linestyle="--", label=f"P99: {percentil_99:.2f} m")

contagens, bins, _ = plt.hist(
    df_series["Hs"].dropna(),
    bins=25,
    color="blue",
    alpha=0.7,
    edgecolor="black",
    label="$H_s$ calculado",
)
topo = contagens.max()

ax = plt.gca()
ax.set_yscale("function", functions=(np.sqrt, np.square))
ax.set_ylim(bottom=0, top=topo * 1.1)
candidatos = [0, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000,
              6000, 8000, 10000, 15000, 20000]
ax.set_yticks([t for t in candidatos if t <= topo * 1.1])

plt.legend()
plt.title(f"Histograma de $H_s$ - Boia {id_boia} ({local})\nPeríodo {periodo}",
          fontsize=11, fontweight="bold")
plt.xlabel("Altura $H_s$ (metros)", fontsize=11)
plt.ylabel("Frequência $N$ (número de registros)\nescala de raiz quadrada ($\u221AN$)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

caminho_img_hs = os.path.join("dados", f"histograma_hs_{id_boia}.png")
plt.savefig(caminho_img_hs, dpi=300, bbox_inches="tight")
plt.show()

# Período de pico
plt.figure(figsize=(7, 5))
plt.hist(df_series["Tp"].dropna(), bins=15, color="green", edgecolor="black", alpha=0.7)

plt.title(f"Histograma de $T_p$ - Boia {id_boia} ({local})\nPeríodo {periodo}",
          fontsize=11, fontweight="bold")
plt.xlabel("Período de Pico $T_p$ (segundos)", fontsize=11)
plt.ylabel("Frequência $N$ (número de registros)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

caminho_img_tp = os.path.join("dados", f"histograma_tp_{id_boia}.png")
plt.savefig(caminho_img_tp, dpi=300, bbox_inches="tight")
plt.show()