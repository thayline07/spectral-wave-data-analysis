import os
import urllib.request
import pandas as pd
import numpy as np
import pickle

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2019, 2020, 2021)
# 46080 (2023, 2024, 2025)

# Configurações
id_boia = "46080"
year = 2025

# Baixar arquivos do NDBC

dir_destino = os.path.join("dados", f"{id_boia}_{year}")

if not os.path.exists(dir_destino):
    os.makedirs(dir_destino)
    print(f"Diretório '{dir_destino}' criado com sucesso!")
else:
    print(f"Diretório '{dir_destino}' já existe.")

regras_dados = {
    "wind": {"pasta_web": "stdmet", "sufixo": "h"},
}

downloads_completos = True


for parametro, config in regras_dados.items():
    pasta_serv = config["pasta_web"]
    letra_sufixo = config["sufixo"]

    nome_arquivo = f"{id_boia}{letra_sufixo}{year}.txt.gz"

    url_completa = f"https://www.ndbc.noaa.gov/data/historical/{pasta_serv}/{nome_arquivo}"

    caminho_salv = os.path.join(dir_destino, nome_arquivo)

    if os.path.exists(caminho_salv):
        print(f"Arquivo '{nome_arquivo}' já existe. Pulando download.")
    else:
        req = urllib.request.Request(url_completa, headers={'User-Agent': 'Mozilla/5.0'})

        try: 
            with urllib.request.urlopen(req) as response:
                with open(caminho_salv, 'wb') as arq_local:
                    arq_local.write(response.read())
            print(f"Arquivo '{parametro}' baixado com sucesso!")

        except urllib.error.HTTPError as e:
            print(
                f" Falha ao acessar [{parametro.upper()}]: Erro HTTP {e.code} - {e.reason}"
            )
            downloads_completos = False
            break

        except Exception as e:
            print(f"Falha ao baixar {parametro.upper()}")
            downloads_completos = False

            if not downloads_completos:
                print(f"Não existem dados espectrais direcionais COMPLETOS para a boia {id_boia} no ano de {year}.")
                break


nome_arq = f"{id_boia}{regras_dados[parametro]['sufixo']}{year}.txt.gz"
caminho_arq = os.path.join(dir_destino, nome_arq)

if os.path.exists(caminho_arq):
    df = pd.read_csv(caminho_arq, sep=r'\s+', skiprows=[1])

df["WDIR"] = df["WDIR"].replace(999, np.nan)
df["WSPD"] = df["WSPD"].replace(99.0, np.nan)

df_series = pd.DataFrame({
    "year": df["#YY"],
    "month": df["MM"],
    "day": df["DD"],
    "hour": df["hh"],
    "minute": df["mm"], 
    "dir": df["WDIR"],
    "vel": df["WSPD"],
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
print(df_series)


dir_destino = os.path.join("dados", f"{id_boia}_{year}")

caminho_csv = os.path.join(dir_destino, f"wind_{id_boia}_{year}.csv")
df_series.to_csv(caminho_csv, index=False)
