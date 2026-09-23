import os
import matplotlib.pyplot as plt
import numpy as np

freq = np.linspace(0.03, 0.4, 1000)
frequencias = np.linspace(0.03, 0.4, 47)

angulos_graus = np.arange(0, 360, 1)
angulos_radianos = np.radians(angulos_graus)

matriz_alvo = np.zeros((len(frequencias), len(angulos_graus)))
matriz_fourier = np.zeros((len(frequencias), len(angulos_graus)))
matriz_mem = np.zeros((len(frequencias), len(angulos_graus)))

direcao_pico_graus = 90.0
direcao_pico_radianos = np.radians(direcao_pico_graus)
s_espalhamento = 4

for f_idx, f in enumerate(frequencias):
    energia_f = np.exp(-((f - 0.1) ** 2) / (2 * (0.03**2)))

    # Função de Espalhamento
    termo_cos = np.cos((angulos_radianos - direcao_pico_radianos) / 2)
    D = termo_cos ** (2 * s_espalhamento)
    D = D / np.sum(D * np.radians(1))

    # Expansão de Fourier
    dtheta = np.radians(1)

    a1 = np.sum(D * np.cos(angulos_radianos) * dtheta)
    b1 = np.sum(D * np.sin(angulos_radianos) * dtheta)

    a2 = np.sum(D * np.cos(2 * angulos_radianos) * dtheta)
    b2 = np.sum(D * np.sin(2 * angulos_radianos) * dtheta)

    r1 = np.sqrt(a1**2 + b1**2)
    r2 = np.sqrt(a2**2 + b2**2)

    alpha1 = np.arctan2(b1, a1)
    alpha2 = np.arctan2(b2, a2) / 2.0

    alpha1_deg = np.degrees(alpha1) % 360
    alpha2_deg = np.degrees(alpha2) % 360

    D_fourier = (1.0 / np.pi) * (
        0.5
        + r1 * np.cos(angulos_radianos - alpha1)
        + r2 * np.cos(2 * (angulos_radianos - alpha2))
    )

    # Reconstrução por Máxima Entropia
    c1 = a1 + (1j * b1)
    c2 = a2 + (1j * b2)

    phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1) ** 2))
    phi1 = c1 - (phi2 * np.conj(c1))

    termo_exp1 = np.exp(-1j * angulos_radianos)
    termo_exp2 = np.exp(-2j * angulos_radianos)

    numerador = 1.0 - np.real(phi1 * np.conj(c1) + phi2 * np.conj(c2))

    denominador = (
        np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2
    )

    D_mem = (1.0 / (2.0 * np.pi)) * (numerador / denominador)
    D_mem = np.real(D_mem)

    matriz_alvo[f_idx, :] = D * energia_f
    matriz_fourier[f_idx, :] = D_fourier * energia_f
    matriz_mem[f_idx, :] = D_mem * energia_f

# Gráficos polares
Angulos, Freqs = np.meshgrid(angulos_radianos, frequencias)

fig, axes = plt.subplots(
    1, 3, figsize=(18, 5), subplot_kw={"projection": "polar"}
)

# Teórico: Alvo
contorno0 = axes[0].contourf(Angulos, Freqs, matriz_alvo, cmap="jet", levels=30)
axes[0].set_title("1. Mar sintético", fontsize=11, fontweight="bold")

# Fourier
contorno1 = axes[1].contourf(
    Angulos, Freqs, matriz_fourier, cmap="jet", levels=30
)
axes[1].set_title("2. Fourier", fontsize=11, fontweight="bold")

# Máxima entropia
contorno2 = axes[2].contourf(Angulos, Freqs, matriz_mem, cmap="jet", levels=30)
axes[2].set_title("3. Máxima entropia", fontsize=11, fontweight="bold")


for ax in axes:
    ax.set_theta_zero_location("N")  
    ax.set_theta_direction(-1)  
    ax.set_ylim([frequencias.min(), frequencias.max()])
    ax.grid(linestyle="--", alpha=0.5)

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(contorno0, cax=cbar_ax, label="Densidade de Energia")

plt.savefig("validacao_espectro_sintetico.png", dpi=300)
plt.show()
