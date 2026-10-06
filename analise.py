import os
import pickle
import numpy as np
import pandas as pd

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2019, 2020, 2021)
# 46080 (2023, 2024, 2025)

id_boia = "41004"
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
    "minute": df_w["mm"], 
    "Hs": lista_hs,
    "Tp": lista_tp,
})

df_series["data_registro"] = pd.to_datetime(
    pd.DataFrame({
        "year": df_series["year"],
        "month": df_series["month"],
        "day": df_series["day"],
        "hour": df_series["hour"],
        "minute": df_series["minute"]
    })
)

df_series = df_series.drop_duplicates(subset=["data_registro"])

df_series = df_series.set_index("data_registro").sort_index()

intervalo_minutos = pd.Series(np.diff(df_series.index)).value_counts().index[0]
intervalo_minutos_int = int(intervalo_minutos.total_seconds() / 60)

data_inicio = df_series.index.min()
data_fim = df_series.index.max()

calendario_completo = pd.date_range(start=data_inicio, end=data_fim, freq=f'{intervalo_minutos_int}min')

df_base = df_series.reindex(calendario_completo)
df_base.index.name = "data_registro"

total_horas_esperadas = len(calendario_completo)
total_horas_registradas = len(df_series)
gaps = df_base["Hs"].isna().sum()

dir_destino = os.path.join("dados", f"{id_boia}_{year}")

caminho_csv = os.path.join(dir_destino, f"series_temporais_{id_boia}_{year}.csv")
df_series.to_csv(caminho_csv)
