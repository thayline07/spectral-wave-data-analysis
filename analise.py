import os
import pickle
import numpy as np
import pandas as pd



id_boia = "46075"
year = 2023
caminho = os.path.join("dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl")

try:
    with open(caminho, 'rb') as f:
        dados = pickle.load(f)
except FileNotFoundError:
    print(f"Arquivo {caminho} não encontrado.")
    exit(1)

# Energia
df_w = dados['w']

frequencias = np.array([float(col) for col in df_w.columns[5:]])

# Largura entre uma frequência e outra
largura_df = np.diff(frequencias)
largura_df = np.append(largura_df, largura_df[-1])

lista_hs = []
lista_tp = []

for index, row in df_w.iterrows():
    energia_linha = row.iloc[5:].values.astype(float)

    if np.isnan(energia_linha).all():
        lista_hs.append(np.nan)
        lista_tp.append(np.nan)
        continue

    m0 = np.sum(energia_linha * largura_df)
    hs = 4 * np.sqrt(m0)

    indice_tp = np.argmax(energia_linha)
    freq_pico = frequencias[indice_tp]
    tp = 1 / freq_pico if freq_pico > 0 else np.nan

    lista_hs.append(hs)
    lista_tp.append(tp)


df_series = pd.DataFrame({
    "year": df_w["#YY"],
    "month": df_w["MM"],
    "day": df_w["DD"],
    "hour": df_w["hh"],
    "Hs": lista_hs,
    "Tp": lista_tp,
})

horas_ausentes = df_series["Hs"].isna().sum()
print(f"Quantidade de horas ausentes: {horas_ausentes}")

df_series["data_registro"] = pd.to_datetime(df_series[["year", "month", "day", "hour"]])

dir_destino = os.path.join("dados", f"{id_boia}_{year}")

caminho_csv = os.path.join(dir_destino, f"series_temporais_{id_boia}_{year}.csv")
df_series.to_csv(caminho_csv)
