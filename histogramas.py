import os 
import matplotlib.pyplot as plt
import pandas as pd

id_boia = "41040"
year = 2023

caminho_csv = os.path.join("dados", f"{id_boia}_{year}", f"series_temporais_{id_boia}_{year}.csv")

if os.path.exists(caminho_csv):
    df_series = pd.read_csv(caminho_csv, parse_dates=["data_registro"])
    df_series = df_series.set_index("data_registro")
else:
    print(f"Arquivo {caminho_csv} não encontrado.")
    exit(1)

# Histogramas

# Altura significativa 
plt.figure(figsize=(7, 5))
plt.hist(
    df_series["Hs"].dropna(),
    bins=25,
    color="blue",
    alpha=0.7,
    edgecolor="black",
)
plt.title(f"Histograma de Hs - Boia {id_boia} - Ano {year}", fontsize=12, fontweight="bold")
plt.xlabel("Altura $H_s$ (metros)", fontsize=11)
plt.ylabel("Frequência (Quantidade de Horas)", fontsize=11)
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