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

plt.plot(freqs, E)
#plt.show()

# ---------------------

angulos_graus = np.arange(0, 361, 1)
angulos_rad = np.radians(angulos_graus)

theta_0 = 270
s = 10 # espalhamento

delta = ((angulos_graus - theta_0 + 180) % 360) - 180

# D:
D = np.cos(np.radians(delta)/2.0)**(2*s)
D = D / np.trapezoid(D, x=angulos_rad)

print(D[angulos_graus == 90])
