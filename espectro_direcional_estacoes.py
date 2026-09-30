import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

id_boia = "46075"
year = 2025

caminho_arq = os.path.join(
    "dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl"
)

if os.path.exists(caminho_arq):
    with open(caminho_arq, "rb") as arq:
        dados = pickle.load(arq)
    print(f"✅ Arquivo '{caminho_arq}' carregado com sucesso!\n")
else:
    print(f"Arquivo '{caminho_arq}' não encontrado.")
    exit(1)

df_w = dados["w"]
df_alpha1 = dados["alpha1"]
df_alpha2 = dados["alpha2"]
df_r1 = dados["r1"]
df_r2 = dados["r2"]

frequencias = np.array([float(col) for col in df_w.columns[5:]])

df_w["data_registro"] = pd.to_datetime(
    pd.DataFrame({
        "year": df_w["#YY"],
        "month": df_w["MM"],
        "day": df_w["DD"],
        "hour": df_w["hh"],
        "minute": df_w["mm"]
    })
)


estacoes_hn = {
    "Inverno":[12, 1, 2],
    "Primavera": [3, 4, 5],
    "Verão": [6, 7, 8],
    "Outono": [9, 10, 11]
}

angulos_graus = np.arange(0, 360, 1)
angulos_rad = np.radians(angulos_graus)


for nome_estacao, meses in estacoes_hn.items():
    
    indices_estacao = df_w[df_w["data_registro"].dt.month.isin(meses)].index
    
    if len(indices_estacao) == 0:
        print(f"Sem dados disponíveis para a estação: {nome_estacao}")
        continue
        
    
    w_medio = df_w.iloc[indices_estacao, 5:-1].mean(axis=0).values.astype(float)
    r1_medio = df_r1.iloc[indices_estacao, 5:].mean(axis=0).values.astype(float)
    r2_medio = df_r2.iloc[indices_estacao, 5:].mean(axis=0).values.astype(float)

    alpha1_dia = df_alpha1.iloc[indices_estacao, 5:].values.astype(float)
    alpha2_dia = df_alpha2.iloc[indices_estacao, 5:].values.astype(float)

    alpha1_medio = np.arctan2(
        np.mean(np.sin(np.radians(alpha1_dia)), axis=0),
        np.mean(np.cos(np.radians(alpha1_dia)), axis=0),
    )
    alpha2_medio = np.arctan2(
        np.mean(np.sin(np.radians(2.0 * alpha2_dia)), axis=0),
        np.mean(np.cos(np.radians(2.0 * alpha2_dia)), axis=0),
    ) / 2.0

    matriz_fourier = np.zeros((len(frequencias), len(angulos_graus)))
    matriz_max_entropia = np.zeros((len(frequencias), len(angulos_graus)))

    for freq in range(len(frequencias)):
        w_f = w_medio[freq]
        r1_f = r1_medio[freq]
        r2_f = r2_medio[freq]
        a1_f = alpha1_medio[freq]
        a2_f = alpha2_medio[freq]

        if np.isnan([w_f, r1_f, r2_f, a1_f, a2_f]).any():
            continue

        # Fourier
        D_fourier = (1.0 / np.pi) * (
            0.5
            + (r1_f * np.cos(angulos_rad - a1_f))
            + (r2_f * np.cos(2.0 * (angulos_rad - a2_f)))
        )
        matriz_fourier[freq, :] = D_fourier * w_f

        # MEM com regularização
        c1 = r1_f * np.exp(1j * a1_f)
        c2 = r2_f * np.exp(1j * 2.0 * a2_f)

        if np.abs(c1) >= 0.99:
            c1 = c1 * 0.99 / np.abs(c1)

        phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1) ** 2))
        phi1 = c1 - (phi2 * np.conj(c1))

        termo_exp1 = np.exp(-1j * angulos_rad)
        termo_exp2 = np.exp(-2j * angulos_rad)

        numerador = 1.0 - np.real(phi1 * np.conj(c1) + phi2 * np.conj(c2))
        denominador = (np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2)
        denominador = np.maximum(denominador, 1e-4)

        D_max_entropia = (1.0 / (2.0 * np.pi)) * (numerador / denominador)
        matriz_max_entropia[freq, :] = D_max_entropia.real * w_f

    #Gráficos 
    Angulos, Freqs = np.meshgrid(angulos_rad, frequencias)

    v_min = np.min(matriz_fourier)
    v_max = np.max(matriz_max_entropia)
    niveis_cores = np.linspace(v_min, v_max, 50)

    fig1, axes1 = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={"projection": "polar"})

    contorno1 = axes1[0].contourf(Angulos, Freqs, matriz_fourier, cmap="jet", levels=niveis_cores)
    axes1[0].set_title(f"Reconstrução por Fourier\n(Média: {nome_estacao})", fontsize=12, fontweight="bold")

    matriz_max_entropia_plot = np.maximum(matriz_max_entropia, 0)

    contorno2 = axes1[1].contourf(Angulos, Freqs, matriz_max_entropia_plot, cmap="jet", levels=niveis_cores)
    axes1[1].set_title(f"Reconstrução por Máxima Entropia (MEM)\n(Média: {nome_estacao})", fontsize=12, fontweight="bold")

    for ax in axes1:
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim([frequencias.min(), frequencias.max()])

    fig1.subplots_adjust(right=0.82, top=0.85, bottom=0.15, wspace=0.3)
    cbar_ax = fig1.add_axes([0.88, 0.15, 0.02, 0.65])
    fig1.colorbar(contorno2, cax=cbar_ax, label="Densidade de Energia ($m^2/Hz/rad$)")

    caminho_polar = f"reconstrucao_polar_{nome_estacao.lower()}_{id_boia}_{year}.png"
    plt.savefig(caminho_polar, dpi=300, bbox_inches="tight")
    plt.close()