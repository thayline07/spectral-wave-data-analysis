import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
import pandas as pd

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2019, 2020, 2021)
# 46080 (2023, 2024, 2025)

id_boia = "41004"
year = 2025

# Dados são registrados nos minutos 10 e 40
data_alvo = pd.Timestamp(f"{year}-01-06 20:10:00")

caminho_arq = os.path.join("dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl")
caminho_wind = os.path.join("dados", f"{id_boia}_{year}", f"wind_{id_boia}_{year}.csv")

if os.path.exists(caminho_arq):
    with open(caminho_arq, 'rb') as arq:
        dados = pickle.load(arq)
    print(f"Arquivo '{caminho_arq}' carregado com sucesso!")
else:
    print(f"Arquivo '{caminho_arq}' não encontrado. Por favor, execute o download primeiro.")
    exit(1)


# Frequências
df_w = dados['w']
frequencias = np.array([float(col) for col in df_w.columns[5:]])
print("Frequências carregadas:", len(frequencias))

df_w["data_registro"] = pd.to_datetime(
    pd.DataFrame({
        "year": df_w["#YY"],
        "month": df_w["MM"],
        "day": df_w["DD"],
        "hour": df_w["hh"],
        "minute": df_w["mm"]
    })
)

linha_alvo = df_w[df_w['data_registro'] == data_alvo].index


if len(linha_alvo) == 0:
    print(f"Erro: A data '{data_alvo}' não foi encontrada no arquivo!")
    exit(1)

# Vento
wdir_vento = wspd_vento = np.nan
hora_vento = None
if os.path.exists(caminho_wind):
    df_wind = pd.read_csv(caminho_wind, parse_dates=["data_registro"])
    df_wind = df_wind.dropna(subset=["dir", "vel"])
    df_wind["hora_vento"] = df_wind["data_registro"]
    df_wind = df_wind.sort_values("data_registro")

    tempos_espectro = df_w[["data_registro"]].sort_values("data_registro")

    df_merged = pd.merge_asof(
        tempos_espectro,
        df_wind[["data_registro", "hora_vento", "dir", "vel"]],
        on="data_registro",
        direction="nearest",
        tolerance=pd.Timedelta("20min")
    )

    linha_vento = df_merged[df_merged["data_registro"] == data_alvo]
    wdir_vento = linha_vento["dir"].values[0]
    wspd_vento = linha_vento["vel"].values[0]
    hora_vento = linha_vento["hora_vento"].values[0]

    
    
else:
    print(f"\nArquivo de vento '{caminho_wind}' não localizado.")




w = df_w.iloc[linha_alvo, 5:-1].values.astype(float).flatten()
alpha1 = dados['alpha1'].iloc[linha_alvo, 5:].values.astype(float).flatten()
alpha2 = dados['alpha2'].iloc[linha_alvo, 5:].values.astype(float).flatten()
r1 = dados['r1'].iloc[linha_alvo, 5:].values.astype(float).flatten()
r2 = dados['r2'].iloc[linha_alvo, 5:].values.astype(float).flatten()

# até 361 somente para preencher a linha vazia de 360° e fechar o gráfico polar 
angulos_graus = np.arange(0, 360, 1)
angulos_rad = np.radians(angulos_graus)

matriz_fourier = np.zeros((len(frequencias), len(angulos_graus)))
matriz_max_entropia = np.zeros((len(frequencias), len(angulos_graus)))

for freq in range(len(frequencias)):
    w_freq = w[freq]
    r1_freq = r1[freq]
    r2_freq = r2[freq]

    a1_rad = np.radians(alpha1[freq])
    a2_rad = np.radians(alpha2[freq])

    if np.isnan([w_freq, r1_freq, r2_freq, a1_rad, a2_rad]).any():
        continue

    D_fourier = (1.0 / np.pi) * (0.5 + (r1_freq * np.cos(angulos_rad - a1_rad)) + (r2_freq * np.cos(2.0 * (angulos_rad - a2_rad))))

    matriz_fourier[freq, :] = D_fourier * w_freq


    # Máxima entropia
    c1 = r1_freq * np.exp(1j * a1_rad)
    c2 = r2_freq * np.exp(1j * 2.0 * a2_rad)

    phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1) ** 2))
    phi1 = c1 - (phi2 * np.conj(c1))

    termo_exp1 = np.exp(-1j * angulos_rad)
    termo_exp2 = np.exp(-2j * angulos_rad)

    numerador = 1.0 - np.real(phi1 * np.conj(c1) + phi2 * np.conj(c2))
    denominador = np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2

    D_max_entropia = (1.0 / (2.0 * np.pi)) * (numerador / denominador)
    D_max_entropia = np.real(D_max_entropia)

    matriz_max_entropia[freq, :] = D_max_entropia * w_freq

# Gráficos polares
Angulos, Freqs = np.meshgrid(angulos_rad, frequencias)

f_pico_idx = np.argmax(w)
f_pico_valor = frequencias[f_pico_idx]


v_min = np.min(matriz_fourier)
v_max = np.max(matriz_max_entropia)
niveis_cores = np.linspace(v_min, v_max, 30)



fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': 'polar'})
fig.suptitle(f"Boia {id_boia} - {data_alvo:%d/%m/%Y %H:%M} UTC", fontsize=15, fontweight="bold", y=0.98,)

# Confirmar com alpha 1:
#print("Valor de alpha1:")
#print(dados['alpha1'].loc[linha_alvo, ".1200"])

def ao_clicar(event):
    if event.inaxes not in axes:
        return
    direcao = np.degrees(event.xdata) % 360
    i = np.argmin(np.abs(frequencias - event.ydata))  
    print(f"{direcao:.0f}° em {frequencias[i]:.4f} Hz  ->  alpha1 = {alpha1[i]:.0f}°")

fig.canvas.mpl_connect("button_press_event", ao_clicar)



# Expansão de Fourier
contorno1 = axes[0].contourf(Angulos, Freqs, matriz_fourier, cmap='jet', levels=niveis_cores)
axes[0].set_title("Reconstrução por Fourier", fontsize=12, fontweight='bold')

# Máxima entropia
contorno2 = axes[1].contourf(Angulos, Freqs, matriz_max_entropia, cmap='jet', levels=niveis_cores)
axes[1].set_title("Reconstrução por Máxima Entropia", fontsize=12, fontweight='bold')

for ax in axes:
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([frequencias.min(), frequencias.max()])

if not np.isnan(wdir_vento):
    theta_vento = np.radians(wdir_vento)
    f_min, f_max = frequencias.min(), frequencias.max()
    for ax in axes:
        ax.annotate(
            "", xy=(theta_vento, f_min + 0.35 * (f_max - f_min)), xytext=(theta_vento, f_max),
            arrowprops=dict(arrowstyle="-|>", color="white", lw=3, mutation_scale=22,
                            path_effects=[pe.withStroke(linewidth=1, foreground="black")]),
        )

    # Legenda indicando que a seta branca é o vento
    seta_legenda = Line2D([], [], linestyle="None", marker=r"$\longrightarrow$", markersize=30,
                          color="white", markeredgecolor="black", markeredgewidth=0.4)
    fig.legend([seta_legenda], ["Vento"], loc="upper left", fontsize=11,
               facecolor="darkblue", labelcolor="white", framealpha=1)

fig.subplots_adjust(right=1.4)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(contorno2, cax=cbar_ax, label='Densidade de Energia ($m^2/Hz/rad$)')

if not np.isnan(wdir_vento):
    texto_vento = (
        f"Vento: {wdir_vento:.0f}° ,  {wspd_vento:.1f} m/s  "
        f",  Registro das {pd.Timestamp(hora_vento):%H:%M}"
    )
else:
    texto_vento = "Vento: sem registro a até 20 min do espectro"

fig.text(0.5, 0.02, texto_vento, ha="center", fontsize=11)


caminho_polar = f"reconstrucao_polar_{id_boia}_{year}.png"
plt.tight_layout()
plt.savefig(caminho_polar, dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(9, 5))

# Isolamos a linha da frequência de pico de cada matriz
# Como as matrizes guardam "D * w", dividimos pela energia total (w_freq)
# para extrair a distribuição direcional D(theta) pura que integra a 1
w_pico = w[f_pico_idx]
D_fourier_pico = matriz_fourier[f_pico_idx, :] / w_pico
D_mem_pico = matriz_max_entropia[f_pico_idx, :] / w_pico


plt.plot(angulos_graus, D_fourier_pico, label="Expansão de Fourier (Baixa Resolução)", color="orange", linewidth=2.5)
plt.plot(angulos_graus, D_mem_pico, label="Máxima Entropia (MEM) (Alta Resolução)", color="green", linestyle="-.", linewidth=2.5)

if not np.isnan(wdir_vento) and not np.isnan(wspd_vento):
    plt.axvline(x=wdir_vento, color="red", linestyle=":", linewidth=2,
                label=f"Vento: {wdir_vento:.0f}° / {wspd_vento:.1f} m/s")


# Configurações estéticas do gráfico cartesiano
plt.title(f"Comparação de $D(\\theta)$ na Frequência de Pico ({f_pico_valor:.3f} Hz)", fontsize=12, fontweight="bold")
plt.xlabel("Direção (Graus)", fontsize=11)
plt.ylabel("Densidade Direcional $D(\\theta)$", fontsize=11)
plt.xlim(0, 360)
plt.xticks(np.arange(0, 361, 45)) 
plt.grid(linestyle="--", alpha=0.5)
plt.legend(fontsize=10, loc="upper right")

#plt.tight_layout()
caminho_cartesiano = f"comparacao_cartesiana_{id_boia}_{year}.png"
#plt.savefig(caminho_cartesiano, dpi=300)
#plt.show()
