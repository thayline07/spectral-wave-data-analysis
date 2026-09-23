import numpy as np
import matplotlib.pyplot as plt

freq = np.linspace(0.03, 0.4, 1000)

angulos_graus = np.arange(0, 360, 1)
angulos_radianos = np.radians(angulos_graus)

direcao_pico_graus = 90.0
direcao_pico_radianos = np.radians(direcao_pico_graus)
s_espalhamento = 4

# Função de Espalhamento
termo_cos = np.cos((angulos_radianos - direcao_pico_radianos)/2)

D = (termo_cos ** (2 * s_espalhamento))
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

D_fourier = (1.0 / np.pi) * (0.5 + r1 * np.cos(angulos_radianos - alpha1) + r2 * np.cos(2 * (angulos_radianos - alpha2)))

plt.figure(figsize=(10, 12))
plt.subplot(2, 1, 1)
plt.xlabel('Ângulo (graus)')
plt.ylabel('Distribuição de Energia')
plt.plot(angulos_graus, D, label='Distribuição Original', color='blue')
plt.subplot(2, 1, 2)
plt.xlabel('Ângulo (graus)')
plt.ylabel('Distribuição de Energia')
plt.plot(angulos_graus, D_fourier, label='Reconstrução via Fourier', color='orange', linestyle='--')
plt.savefig('reconstrucao_fourier.png', dpi=300)


# Reconstrução por Máxima Entropia

c1 = a1 + (1j * b1)
c2 = a2 + (1j * b2)

phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1)**2))
phi1 = c1 - (phi2 * np.conj(c1))

termo_exp1 = np.exp(-1j * angulos_radianos)
termo_exp2 = np.exp(-2j * angulos_radianos)

numerador = 1.0 - (phi1 * np.conj(c1) + phi2 * np.conj(c2))

denominador = np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2

D_mem = (1.0 / (2.0 * np.pi)) * (numerador / denominador)

D_mem = np.real(D_mem)

plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plt.plot(angulos_graus, D, label='Distribuição Original', color='blue', linewidth=2)
plt.ylabel('Distribuição de Energia')
plt.title('Espectro Sintético: Original vs. máxima entropia', fontsize=12, fontweight='bold')
plt.xlim(0, 360)
plt.grid(linestyle="--", alpha=0.5)

plt.subplot(2, 1, 2)
plt.plot(angulos_graus, D_mem, label='Reconstrução via máxima entropia', color='green', linestyle='-.', linewidth=2)
plt.xlabel('Ângulo (graus)')
plt.ylabel('Distribuição de Energia')
plt.xlim(0, 360)
plt.grid(linestyle="--", alpha=0.5)
plt.savefig('reconstrucao_mem.png', dpi=300, bbox_inches='tight')
plt.show()
