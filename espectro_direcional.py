import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

id_boia = "46075"
year = 2023

caminho_arq = os.path.join("dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl")

if os.path.exists(caminho_arq):
    with open(caminho_arq, 'rb') as arq:
        dados = pickle.load(arq)
    print(f"Arquivo '{caminho_arq}' carregado com sucesso!")
else:
    print(f"Arquivo '{caminho_arq}' não encontrado. Por favor, execute o download primeiro.")

# Frequências
df_w = dados['w']
frequencias = np.array([float(col) for col in df_w.columns[5:]])
print("Frequências carregadas:", len(frequencias))

w = df_w.iloc[0, 5:].values.astype(float)
alpha1 = dados['alpha1'].iloc[0, 5:].values.astype(float)
alpha2 = dados['alpha2'].iloc[0, 5:].values.astype(float)
r1 = dados['r1'].iloc[0, 5:].values.astype(float)
r2 = dados['r2'].iloc[0, 5:].values.astype(float)

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

fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': 'polar'})

# Expansão de Fourier
contorno1 = axes[0].contourf(Angulos, Freqs, matriz_fourier, cmap='jet', levels=30)
fig.colorbar(contorno1, ax=axes[0], orientation='vertical', label='Densidade de energia (m²/Hz)')
axes[0].set_title("Reconstrução por Fourier", fontsize=12, fontweight='bold')

# Máxima entropia
contorno2 = axes[1].contourf(Angulos, Freqs, matriz_max_entropia, cmap='jet', levels=30)
fig.colorbar(contorno2, ax=axes[1], orientation='vertical', label='Densidade de energia (m²/Hz)')
axes[1].set_title("Reconstrução por Máxima Entropia", fontsize=12, fontweight='bold')

for ax in axes:
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([frequencias.min(), frequencias.max()])

plt.tight_layout()

plt.savefig(f"reconstrucao_{id_boia}_{year}.png", dpi=300)
plt.show()

"""
==============================================================================
ANÁLISE E COMPARAÇÃO DOS MÉTODOS DE RECONSTRUÇÃO DIRECIONAL (BOIA 46075)
==============================================================================

A partir dos gráficos polares gerados para a primeira hora do ano de 2023, 
na estação 46075 (Alasca), a comparação entre os métodos evidencia o seguinte:

1. COMPORTAMENTO FÍSICO E GEOGRÁFICO DO MAR REAL:
   - Ambas as reconstruções identificam com sucesso que o sistema de ondas 
     dominante para este horário vem da direção de Leste (concentrado entre 
     80° e 100°).
   - O pico de energia ocorre em frequências intermediárias (em torno de 
     0.12 Hz a 0.15 Hz), o que equivale a ondas com períodos de 7 a 8 segundos 
     (caracterizando vagas bem desenvolvidas ou marulhos curtos).

2. EXPANSÃO DE FOURIER (TRUNCADA EM 2ª ORDEM):
   - Apresenta uma resolução angular visivelmente mais baixa, gerando uma 
     mancha de energia muito mais "esparramada" (borrada) ao redor do círculo.
   - Restrição Matemática/Física: Por utilizar uma série truncada, o método 
     sofre com oscilações numéricas e gera valores de densidade de energia 
     NEGATIVOS (conforme indicado na escala da barra de cores em -0.24), o que 
     é fisicamente impossível no oceano real.

3. MÉTODO DA MÁXIMA ENTROPIA (MEM):
   - Exibe uma resolução angular infinitamente superior. A mancha de energia 
     é estreita, nítida e muito mais focada na direção real de propagação.
   - Vantagem Física: O MEM respeita estritamente a conservação de energia e 
     as leis da física, travando o limite inferior da escala de densidade 
     estritamente em 0.00 (eliminando energias negativas artificiais).

CONCLUSÃO DA ENTREGA:
O Método da Máxima Entropia (MEM) valida-se como a ferramenta mais robusta 
e precisa para o mapeamento direcional do espectro de ondas, sendo ideal 
para separar sistemas complexos e focados em mar aberto onde a Expansão de 
Fourier falha por excesso de suavização espacial.
==============================================================================
"""
