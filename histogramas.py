import os 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

id_boia = "46080"
year = 2025

caminho_csv = os.path.join("dados", f"{id_boia}_{year}", f"series_temporais_{id_boia}_{year}.csv")

if os.path.exists(caminho_csv):
    df_series = pd.read_csv(caminho_csv, parse_dates=["data_registro"])
    df_series = df_series.set_index("data_registro")
else:
    print(f"Arquivo {caminho_csv} não encontrado.")
    exit(1)

# Histogramas

# Altura significativa 
altura_máxima = df_series["Hs"].max()
mediana_hs = df_series["Hs"].median()
percentil_99 = df_series["Hs"].quantile(0.99)
print(mediana_hs, altura_máxima, percentil_99)

def checar_vizinhos(df, valor_max, n_vizinhos=5):
    hs = df["Hs"]
    pos = (hs - valor_max).abs().to_numpy().argmin()

    inicio = max(pos - n_vizinhos, 0)
    fim = pos + n_vizinhos + 1

    return df.iloc[inicio:fim][["Hs"]]

print(f"\nVizinhança do Hs máximo (boia {id_boia}, ano: {year}):")
print(checar_vizinhos(df_series, altura_máxima))



plt.figure(figsize=(7, 5))
plt.axvline(x=mediana_hs, color='red', linestyle='--', label=f'Mediana: {mediana_hs:.2f} m')
plt.axvline(x=altura_máxima, color='green', linestyle='--', label=f'Hs Máximo: {altura_máxima:.2f} m')
plt.axvline(x=percentil_99, color='orange', linestyle='--', label=f'P99: {percentil_99:.2f} m')
plt.hist(
    df_series["Hs"].dropna(),
    bins=25,
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
ax = plt.gca()
ax.set_yscale('function', functions=(np.sqrt, np.square))
ax.set_ylim(bottom=0)
ax.set_yticks([0, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000])

plt.legend()
plt.title(f"Histograma de Hs - Boia {id_boia} - Ano {year}", fontsize=12, fontweight="bold")
plt.xlabel("Altura $H_s$ (metros)", fontsize=11)
plt.ylabel("Frequência (Número de registros)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()


caminho_img_hs = os.path.join(
    "dados", f"{id_boia}_{year}", f"histograma_hs_{id_boia}_{year}.png"
)
plt.savefig(caminho_img_hs, dpi=300)
plt.show()

# Período de pico
plt.figure(figsize=(7, 5))
plt.hist(df_series["Tp"], bins=15, color="green", edgecolor="black", alpha=0.7)

plt.title(f"Distribuição Período de Pico - Boia {id_boia} - Ano {year}", fontsize=12, fontweight="bold")

plt.xlabel("Período de Pico $T_p$ (segundos)", fontsize=11)
plt.ylabel("Frequência (Quantidade de Horas)", fontsize=11)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()

caminho_img_tp = os.path.join(
    "dados", f"{id_boia}_{year}", f"histograma_tp_{id_boia}_{year}.png"
)
plt.savefig(caminho_img_tp, dpi=300)
plt.show()
