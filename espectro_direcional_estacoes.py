import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from download_arquivos import baixar_dados

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2019, 2020, 2021)
# 46080 (2023, 2024, 2025)

id_boia = "46080"
year = 2024

def carregar_dados(id_boia, year):
    caminho_arq = os.path.join(
        "dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl"
    )

    if not os.path.exists(caminho_arq):
        print(f"Arquivo '{caminho_arq}' não encontrado. Tentando baixar...")
        caminho_arq = baixar_dados(id_boia, year)
        if caminho_arq is None:
            return None

    with open(caminho_arq, "rb") as arq:
        dados = pickle.load(arq)
    print(f"Arquivo '{caminho_arq}' carregado com sucesso!\n")
    return dados


def filtrar_meses(df, ano, meses):
    return df[(df["#YY"] == ano) & (df["MM"].isin(meses))]


dados = carregar_dados(id_boia, year)
if dados is None:
    exit(1)

# Dezembro do ano anterior
dados_ano_anterior = carregar_dados(id_boia, year - 1)

parametros = ["w", "alpha1", "alpha2", "r1", "r2"]

frequencias = np.array([float(col) for col in dados["w"].columns[5:]])


estacoes_hn = {
    "Inverno":[12, 1, 2],
    "Primavera": [3, 4, 5],
    "Verão": [6, 7, 8],
    "Outono": [9, 10, 11]
}

angulos_graus = np.arange(0, 360, 1)
angulos_rad = np.radians(angulos_graus)


for nome_estacao, meses in estacoes_hn.items():

    if nome_estacao == "Inverno":
        dados_estacao = {p: filtrar_meses(dados[p], year, [1, 2]) for p in parametros}

        if dados_ano_anterior is not None:
            for p in parametros:
                dezembro = filtrar_meses(dados_ano_anterior[p], year - 1, [12])
                dezembro = dezembro.reindex(columns=dados[p].columns)
                dados_estacao[p] = pd.concat([dezembro, dados_estacao[p]], ignore_index=True)
            rotulo = f"Inverno dez/{year - 1} a fev/{year}"
        else:
            print(f"Sem dados de {year - 1}: o inverno usará só jan e fev de {year}.")
            rotulo = f"Inverno jan a fev/{year} (sem dez/{year - 1})"
    else:
        dados_estacao = {p: filtrar_meses(dados[p], year, meses) for p in parametros}
        rotulo = nome_estacao

    if len(dados_estacao["w"]) == 0:
        print(f"Sem dados disponíveis para a estação: {nome_estacao}")
        continue

    w_est = dados_estacao["w"].iloc[:, 5:].values.astype(float)
    r1_est = dados_estacao["r1"].iloc[:, 5:].values.astype(float)
    r2_est = dados_estacao["r2"].iloc[:, 5:].values.astype(float)
    alpha1_est = np.radians(dados_estacao["alpha1"].iloc[:, 5:].values.astype(float))
    alpha2_est = np.radians(dados_estacao["alpha2"].iloc[:, 5:].values.astype(float))

    c1_est = r1_est * np.exp(1j * alpha1_est)
    c2_est = r2_est * np.exp(2j * alpha2_est)

    validos = ~(np.isnan(w_est) | np.isnan(c1_est) | np.isnan(c2_est))
    peso = np.where(validos, w_est, 0.0)
    soma_pesos = peso.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        c1_medio = np.sum(peso * np.where(validos, c1_est, 0), axis=0) / soma_pesos
        c2_medio = np.sum(peso * np.where(validos, c2_est, 0), axis=0) / soma_pesos

    w_medio = np.nanmean(w_est, axis=0)

    matriz_fourier = np.zeros((len(frequencias), len(angulos_graus)))
    matriz_max_entropia = np.zeros((len(frequencias), len(angulos_graus)))

    for freq in range(len(frequencias)):
        w_f = w_medio[freq]
        c1 = c1_medio[freq]
        c2 = c2_medio[freq]

        r1 = np.abs(c1)
        a1 = np.angle(c1)
        r2 = np.abs(c2)
        a2 = np.angle(c2)/2.0

        if np.isnan([w_f, r1, r2, a1, a2]).any():
            continue

        # Fourier
        D_fourier = (1.0 / np.pi) * (
            0.5
            + (r1 * np.cos(angulos_rad - a1))
            + (r2 * np.cos(2.0 * (angulos_rad - a2)))
        )
        matriz_fourier[freq, :] = D_fourier * w_f

        if r1 >= 1.0:
            print(f"{frequencias[freq]:.3f} Hz: |c1| = 1, MEM indefinido. Pulando.")
            continue

        phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1) ** 2))
        phi1 = c1 - (phi2 * np.conj(c1))

        termo_exp1 = np.exp(-1j * angulos_rad)
        termo_exp2 = np.exp(-2j * angulos_rad)

        numerador = 1.0 - np.real(phi1 * np.conj(c1) + phi2 * np.conj(c2))
        denominador = (np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2)

        D_max_entropia = (1.0 / (2.0 * np.pi)) * (numerador / denominador)
        matriz_max_entropia[freq, :] = D_max_entropia.real * w_f

    #Gráficos 
    Angulos, Freqs = np.meshgrid(angulos_rad, frequencias)

    v_min = np.min(matriz_fourier)
    v_max = np.max(matriz_max_entropia)
    niveis_cores = np.linspace(v_min, v_max, 50)

    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"projection": "polar"})

    contorno1 = axes1[0].contourf(Angulos, Freqs, matriz_fourier, cmap="jet", levels=niveis_cores)
    axes1[0].set_title(f"Reconstrução por Fourier\n(Média: {rotulo})", fontsize=12, fontweight="bold")


    contorno2 = axes1[1].contourf(Angulos, Freqs, matriz_max_entropia, cmap="jet", levels=niveis_cores)
    axes1[1].set_title(f"Reconstrução por Máxima Entropia (MEM)\n(Média: {rotulo})", fontsize=12, fontweight="bold")

    for ax in axes1:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim([frequencias.min(), frequencias.max()])

    fig1.subplots_adjust(right=0.82, top=0.85, bottom=0.15, wspace=0.3)
    cbar_ax = fig1.add_axes([0.88, 0.15, 0.02, 0.65])
    fig1.colorbar(contorno2, cax=cbar_ax, label="Densidade de Energia ($m^2/Hz/rad$)")
    plt.show()

    caminho_polar = f"reconstrucao_polar_{nome_estacao.lower()}_{id_boia}_{year}.png"
    #plt.savefig(caminho_polar, dpi=300, bbox_inches="tight")
    plt.close()
