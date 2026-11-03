import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Validação da reconstrução com espectros sintéticos

id_boia = "41040"
year = 2019

# Pegando as 47 frequências reais do ndbc
caminho_pkl = os.path.join("dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl")
if os.path.exists(caminho_pkl):
    with open(caminho_pkl, "rb") as arq:
        freqs = np.array([float(c) for c in pickle.load(arq)["w"].columns[5:]])
if freqs is None:
    print("Nenhum .pkl encontrado para obter as frequências.")
    exit(1)

# Parâmetros escolhidos
Hs = 2.0  
Tp = 10.0
gamma = 3.3

# Função do espectro de JONSWAP
def jonswap(f, Hs, Tp, gamma):
    g = 9.81
    fp = 1 / Tp

    sigma = np.where(f <= fp, 0.07, 0.09)

    alpha = 1

    E_pm = alpha*((g**2)/(((2*np.pi)**4)*f**5))*np.exp((-5/4)*((fp/f)**4))
    r = np.exp(-((f-fp)**2)/(2*(sigma**2)*(fp**2)))
    E_temp = E_pm*(gamma**r)

    m0_inicial = np.trapezoid(E_temp, x=f)
    m0_desejado = (Hs/4)**2

    return E_temp * (m0_desejado/m0_inicial)

E = jonswap(freqs, Hs, Tp, gamma)

m0 = np.trapezoid(E, x=freqs)
print("Hs:" , 4*np.sqrt(m0))
print("m0:", m0)

s_max = np.argmax(E)
print("Freq:", freqs[s_max])

# ---------------------

angulos_graus = np.arange(0, 361, 1)
angulos_rad = np.radians(angulos_graus)

def espalhamento(theta_0, s):
    delta = ((angulos_graus - theta_0 + 180) % 360) - 180
    D = np.cos(np.radians(delta)/2.0)**(2*s)
    D = D / np.trapezoid(D, x=angulos_rad)

    return D

def momentos(D):
    c1 = np.trapezoid(D * np.exp(1j * angulos_rad), angulos_rad)
    c2 = np.trapezoid(D * np.exp(2j * angulos_rad), angulos_rad)

    r1 = np.abs(c1)
    r2 = np.abs(c2)

    a1 = np.degrees(np.angle(c1)) % 360
    a2 = np.degrees(np.angle(c2)) % 360
    return c1, c2, r1, r2, a1, a2

theta_0 = 270

for s in [2, 10, 40]:
    D = espalhamento(theta_0, s)
    c1, c2, r1, r2, a1, a2 = momentos(D)

    print(f"s={s}:  r1={r1:.6f}, r2={r2:.6f}, a1={a1:.6f}, a2={a2:.6f}")


# reconstrução por fourier
D_fourier = np.zeros((len(freqs), len(angulos_rad)))
D_fourier += (1/np.pi)*((1/2) + (r1*np.cos(angulos_rad - np.radians(a1))) + (r2*np.cos(2*(angulos_rad - np.radians(a2)))))

D_mem_lygre_krogstad = np.zeros((len(freqs), len(angulos_rad)))
phi2 = (c2 - (c1**2)) / (1 - (np.abs(c1)**2))
phi1 = c1 - (phi2 * np.conj(c1))
D_mem_lygre_krogstad = (1/(2*np.pi)) * ((1 - np.real((phi1*np.conj(c1)+(phi2*np.conj(c2))))) / np.abs(1 - (phi1*np.exp(-1j*angulos_rad)) - (phi2*np.exp(-2j*angulos_rad)))**2)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': 'polar'})
Angulos, Freqs = np.meshgrid(angulos_rad, freqs)

v_min = np.min(D_fourier)
v_max = np.max(D_fourier)
niveis_cores = np.linspace(v_min, v_max, 30)

contorno1 = axes[0].contourf(Angulos, Freqs, D_fourier, cmap='jet', levels=niveis_cores)
axes[0].set_title("Reconstrução por Fourier", fontsize=12, fontweight='bold')

# Máxima entropia
contorno2 = axes[1].contourf(Angulos, Freqs, D_mem_lygre_krogstad, cmap='jet', levels=niveis_cores)
#axes[1].set_title("Reconstrução por Máxima Entropia", fontsize=12, fontweight='bold')

for ax in axes:
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([freqs.min(), freqs.max()])


fig.subplots_adjust(right=1.4)
cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
fig.colorbar(contorno1, cax=cbar_ax, label='Densidade de Energia ($m^2/Hz/$)')


caminho_polar = f"reconstrucao_polar_{id_boia}_{year}.png"
plt.tight_layout()
plt.show()
