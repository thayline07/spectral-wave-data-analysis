import os
import urllib.request
import pandas as pd
import numpy as np
import pickle

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2023, 2024, 2025)
# 46075 (2023, 2024, 2025)

# Configurações
id_boia = "46075"
year = 2023

# Baixar arquivos do NDBC

dir_destino = os.path.join("dados", f"{id_boia}_{year}")

if not os.path.exists(dir_destino):
    os.makedirs(dir_destino)
    print(f"Diretório '{dir_destino}' criado com sucesso!")
else:
    print(f"Diretório '{dir_destino}' já existe. Nenhuma ação necessária.")

regras_dados = {
    "w": {"pasta_web": "swden", "sufixo": "w"},
    "alpha1": {"pasta_web": "swdir", "sufixo": "d"},
    "alpha2": {"pasta_web": "swdir2", "sufixo": "i"},
    "r1": {"pasta_web": "swr1", "sufixo": "j"},
    "r2": {"pasta_web": "swr2", "sufixo": "k"},
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

# Controle de Qualidade
dados = {}

for parametro in regras_dados.keys():
    nome_arq = f"{id_boia}{regras_dados[parametro]['sufixo']}{year}.txt.gz"
    caminho_arq = os.path.join(dir_destino, nome_arq)

    df = pd.read_csv(caminho_arq, sep=r'\s+', skiprows=[1])

    colunas_dados = df.columns[5:]

    #df[colunas_dados] = df[colunas_dados].apply(pd.to_numeric, errors='coerce')
    df[colunas_dados] = df[colunas_dados].replace([999, 999.0, 99.0, 99.00], np.nan)

    if parametro in ['r1', 'r2']:
        # Correção: Os arquivos de energia espectral (r1 e r2) estão em centésimos, então precisamos multiplicar por 0.01 para obter os valores corretos.
        df[colunas_dados] = df[colunas_dados] * 0.01

    dados[parametro] = df


# Salvar os dataframes em um arquivo
if dados:
    caminho_salvar = os.path.join(dir_destino, f"dados_{id_boia}_{year}.pkl")

    with open(caminho_salvar, 'wb') as f:
        pickle.dump(dados, f)


