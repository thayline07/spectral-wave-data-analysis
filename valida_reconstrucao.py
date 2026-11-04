import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Validação da reconstrução com espectros sintéticos

id_boia = "41040"
year = 2019

# Pegando as 47 frequências reais do ndbc
caminho_pkl = os.path.join("dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl")
freqs = None
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
    a2 = np.degrees(np.angle(c2)/2.0) % 360
    return c1, c2, r1, r2, a1, a2

theta_0 = 270

s = 3
D = espalhamento(theta_0, s)
c1, c2, r1, r2, a1, a2 = momentos(D)

print(f"s={s}:  r1={r1:.6f}, r2={r2:.6f}, a1={a1:.6f}, a2={a2:.6f}")


# reconstrução por fourier
D_fourier = np.zeros((len(freqs), len(angulos_rad)))
D_fourier += (1/np.pi)*((1/2) + (r1*np.cos(angulos_rad - np.radians(a1))) + (r2*np.cos(2*(angulos_rad - np.radians(a2)))))
S_fourier = E[:, None] * D_fourier * (np.pi/180.0)

# MEM de Lygre e Krogstad
D_mem_lygre_krogstad = np.zeros((len(freqs), len(angulos_rad)))
phi2 = (c2 - (c1**2)) / (1 - (np.abs(c1)**2))
phi1 = c1 - (phi2 * np.conj(c1))
D_mem_lygre_krogstad = (1/(2*np.pi)) * ((1 - np.real((phi1*np.conj(c1)+(phi2*np.conj(c2))))) / np.abs(1 - (phi1*np.exp(-1j*angulos_rad)) - (phi2*np.exp(-2j*angulos_rad)))**2)
S_mem_lygre_krogstad = E[:, None] * D_mem_lygre_krogstad * (np.pi/180.0)

# MEM de Kobune e Hashimoto
dth = angulos_rad[1] - angulos_rad[0]
phi = np.array([np.cos(angulos_rad), np.sin(angulos_rad), np.cos(2 * angulos_rad), np.sin(2 * angulos_rad)])

def _dist(lam):
    logw = -(lam @ phi)
    mx = logw.max()
    ew = np.exp(logw - mx)
    Z = ew.sum() * dth
    return ew / Z, np.log(Z) + mx

def mem_shannon(m, tol=1e-10, maxiter=200):
    lam = np.zeros(4)
    D, logZ = _dist(lam)
    F = logZ + lam @ m

    for _ in range(maxiter):
        E = phi @ D * dth
        grad = m - E
        if np.max(np.abs(grad)) < tol:
            return D, True
    
        cov =  (phi * D) @ phi.T * dth - np.outer(E, E)
        try:
            passo = np.linalg.solve(cov + 1e-14 * np.eye(4), grad)
        except np.linalg.LinAlgError:
            return D, False
        t = 1.0
        while t > 1e-12:
            lam_novo = lam - t * passo
            D_novo, logZ_novo = _dist(lam_novo)
            F_novo = logZ_novo + lam_novo @ m
    
            if np.isfinite(F_novo) and F_novo <= F - 1e-4 * t * (grad @ passo):
                break
            t *= 0.5
        else: 
            return D, False
        lam, D, F = lam_novo, D_novo, F_novo
    return D, bool(np.max(np.abs(m - phi @ D * dth)) < 1e-6)

m = np.array([c1.real, c1.imag, c2.real, c2.imag])
viavel = (np.abs(c1) < 1.0) and (np.abs(c2 - c1**2) < 1.0 - np.abs(c1)**2)

D_shannon, ok = mem_shannon(m)

S_shannon = E[:, None] * D_shannon[None, :] * (np.pi/180.0)

S_v = E[:, None] * D[None, :] * (np.pi/180.0)

fig, axes = plt.subplots(2, 2, figsize=(12, 11), subplot_kw={'projection': 'polar'}, layout="constrained")
axes = axes.ravel()
Angulos, Freqs = np.meshgrid(angulos_rad, freqs)

niveis = np.concatenate([S_v.ravel(), S_fourier.ravel(), S_mem_lygre_krogstad.ravel(), S_shannon.ravel()])
niveis_cores = np.linspace(np.nanmin(niveis), np.nanmax(niveis), 30)

contorno0 = axes[0].contourf(Angulos, Freqs, S_v, cmap='jet', levels=niveis_cores)
axes[0].set_title("Espectro Direcional Original", fontsize=12, fontweight='bold')

contorno1 = axes[1].contourf(Angulos, Freqs, S_fourier, cmap='jet', levels=niveis_cores)
axes[1].set_title("Reconstrução por Fourier", fontsize=12, fontweight='bold')

# Máxima entropia
contorno2 = axes[2].contourf(Angulos, Freqs, S_mem_lygre_krogstad, cmap='jet', levels=niveis_cores)
axes[2].set_title("Reconstrução MEM\n(Lygre-Krogstad)", fontsize=12, fontweight='bold')

# MEM de Kobune e Hashimoto
contorno3 = axes[3].contourf(Angulos, Freqs, S_shannon, cmap='jet', levels=niveis_cores)
axes[3].set_title("Reconstrução MEM\n(Kobune e Hashimoto)", fontsize=12, fontweight='bold')

for ax in axes:
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([freqs.min(), freqs.max()])


fig.colorbar(contorno0, ax=axes, shrink=0.75, pad=0.04,
             label="Densidade de Energia ($m^2/Hz/grau$)")


caminho_polar = f"sintetico_polar_s{s}.png"
#plt.tight_layout()
#fig.savefig(caminho_polar, dpi=300, bbox_inches="tight")
plt.show()


# RMSE de cada método contra a original
rmse_fourier = np.sqrt(np.mean((D_fourier[s_max] - D)**2))
rmse_lk      = np.sqrt(np.mean((D_mem_lygre_krogstad - D)**2))
rmse_hk      = np.sqrt(np.mean((D_shannon - D)**2))

rmse_fourier_g = rmse_fourier * np.pi/180
rmse_lk_g      = rmse_lk * np.pi/180
rmse_hk_g      = rmse_hk * np.pi/180

plt.figure(figsize=(9, 5))
plt.plot(angulos_graus, D * np.pi/180, label="Original", color="black", linewidth=2.5)
plt.plot(angulos_graus, D_fourier[s_max] * np.pi/180,
         label=f"Fourier (RMSE = {rmse_fourier_g*1e3:.2f})", color="orange")
plt.plot(angulos_graus, D_mem_lygre_krogstad * np.pi/180,
         label=f"MEM Lygre-Krogstad (RMSE = {rmse_lk_g*1e3:.2f})", color="green", linestyle="-.")
plt.plot(angulos_graus, D_shannon * np.pi/180,
         label=f"MEM Kobune-Hashimoto (RMSE = {rmse_hk_g*1e3:.2f})", color="purple", linewidth=2)
plt.xlabel("Direção (graus)"); plt.ylabel("D (1/grau)")
plt.xlim(0, 360); 
plt.grid(alpha=0.5)
plt.title(f"Distribuição direcional ($\\cos^{{2s}}$, $s={s}$, $\\theta_0={theta_0}°$)", fontsize=12, fontweight="bold")
plt.legend(title="RMSE em unidades de $10^{-3}$ grau$^{-1}$", fontsize=10)
#plt.savefig(f"sintetico_cartesiano_s{s}.png", dpi=300, bbox_inches="tight"); 
plt.show()

# Sistema bimodal

# ===================== PARTE 2: dois sistemas (marulho + vaga) =====================

# Marulho
E1 = jonswap(freqs, 1.5, 11.0, 3.3)
D1 = espalhamento(200, 20)

# Vaga
E2 = jonswap(freqs, 1.5, 9.0, 3.3)
D2 = espalhamento(290, 8)

E_tot = E1 + E2
S_tot = E1[:, None] * D1[None, :] + E2[:, None] * D2[None, :]    # m²/Hz/rad
D_ver = S_tot / E_tot[:, None]                                     # D verdadeiro de cada frequência (1/rad)

print("integral de S_tot em theta == E_tot:",
      np.allclose(np.trapezoid(S_tot, angulos_rad, axis=1), E_tot))
print("Hs total:", 4*np.sqrt(np.trapezoid(E_tot, freqs)), "(esperado ~1,80)")

# ---- reconstrução em cada frequência, só com c1 e c2 ----
def fourier_c(c1, c2):
    r1, a1 = np.abs(c1), np.angle(c1)
    r2, a2 = np.abs(c2), np.angle(c2) / 2.0
    return (0.5 + r1*np.cos(angulos_rad - a1) + r2*np.cos(2*(angulos_rad - a2))) / np.pi

def lygre_c(c1, c2):
    phi2 = (c2 - c1**2) / (1 - np.abs(c1)**2)
    phi1 = c1 - phi2 * np.conj(c1)
    num = 1 - np.real(phi1*np.conj(c1) + phi2*np.conj(c2))
    den = np.abs(1 - phi1*np.exp(-1j*angulos_rad) - phi2*np.exp(-2j*angulos_rad))**2
    return num / (2*np.pi*den)

n = len(freqs)
rec_f  = np.zeros((n, len(angulos_rad)))
rec_lk = np.zeros((n, len(angulos_rad)))
rec_hk = np.zeros((n, len(angulos_rad)))
viavel2 = np.zeros(n, dtype=bool)
conv2   = np.zeros(n, dtype=bool)

for i in range(n):
    c1_i, c2_i = momentos(D_ver[i])[:2]
    viavel2[i] = (np.abs(c1_i) < 1.0) and (np.abs(c2_i - c1_i**2) < 1.0 - np.abs(c1_i)**2)
    rec_f[i]  = fourier_c(c1_i, c2_i)
    rec_lk[i] = lygre_c(c1_i, c2_i)
    m_i = np.array([c1_i.real, c1_i.imag, c2_i.real, c2_i.imag])
    rec_hk[i], conv2[i] = mem_shannon(m_i)

print("todas viáveis:", viavel2.all(), "| HK convergiu em todas:", conv2.all())

# ---- mapas polares (mesmo estilo do caso de um sistema) ----
S_v2  = S_tot * (np.pi/180.0)
S_f2  = E_tot[:, None] * rec_f  * (np.pi/180.0)
S_lk2 = E_tot[:, None] * rec_lk * (np.pi/180.0)
S_hk2 = E_tot[:, None] * rec_hk * (np.pi/180.0)

f_max_plot = 0.3     # acima disso a energia é desprezível; só aproxima o gráfico

fig, axes = plt.subplots(2, 2, figsize=(12, 11), subplot_kw={'projection': 'polar'}, layout="constrained")
axes = axes.ravel()
Angulos, Freqs = np.meshgrid(angulos_rad, freqs)

niveis2 = np.concatenate([S_v2.ravel(), S_f2.ravel(), S_lk2.ravel(), S_hk2.ravel()])
niveis_cores2 = np.linspace(np.nanmin(niveis2), np.nanmax(niveis2), 30)

contorno0 = axes[0].contourf(Angulos, Freqs, S_v2,  cmap='jet', levels=niveis_cores2)
axes[0].set_title("Espectro Direcional Original", fontsize=12, fontweight='bold')
axes[1].contourf(Angulos, Freqs, S_f2,  cmap='jet', levels=niveis_cores2)
axes[1].set_title("Reconstrução por Fourier", fontsize=12, fontweight='bold')
axes[2].contourf(Angulos, Freqs, S_lk2, cmap='jet', levels=niveis_cores2)
axes[2].set_title("Reconstrução MEM\n(Lygre-Krogstad)", fontsize=12, fontweight='bold')
axes[3].contourf(Angulos, Freqs, S_hk2, cmap='jet', levels=niveis_cores2)
axes[3].set_title("Reconstrução MEM\n(Kobune e Hashimoto)", fontsize=12, fontweight='bold')

for ax in axes:
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim([freqs.min(), f_max_plot])

fig.colorbar(contorno0, ax=axes, shrink=0.75, pad=0.04,
             label="Densidade de Energia ($m^2/Hz/grau$)")
fig.savefig("sintetico_dois_sistemas_polar.png", dpi=300, bbox_inches="tight")
plt.show()

# ---- cartesiano: um gráfico por frequência, mesmo estilo do anterior ----
i1, i2 = np.argmax(E1), np.argmax(E2)
lo, hi = sorted((i1, i2))
i_ov = lo + np.argmin(np.abs(E1[lo:hi+1] - E2[lo:hi+1]))     # onde os dois sistemas têm energia parecida

def grafico_cartesiano(i, nome, arquivo):
    D_o = D_ver[i]
    rmse_f_g  = np.sqrt(np.mean((rec_f[i]  - D_o)**2)) * np.pi/180
    rmse_lk_g = np.sqrt(np.mean((rec_lk[i] - D_o)**2)) * np.pi/180
    rmse_hk_g = np.sqrt(np.mean((rec_hk[i] - D_o)**2)) * np.pi/180

    plt.figure(figsize=(9, 5))
    plt.plot(angulos_graus, D_o * np.pi/180, label="Original", color="black", linewidth=2.5)
    plt.plot(angulos_graus, rec_f[i] * np.pi/180,
             label=f"Fourier (RMSE = {rmse_f_g*1e3:.2f})", color="orange")
    plt.plot(angulos_graus, rec_lk[i] * np.pi/180,
             label=f"MEM Lygre-Krogstad (RMSE = {rmse_lk_g*1e3:.2f})", color="green", linestyle="-.")
    plt.plot(angulos_graus, rec_hk[i] * np.pi/180,
             label=f"MEM Kobune-Hashimoto (RMSE = {rmse_hk_g*1e3:.2f})", color="purple", linewidth=2)
    plt.title(f"{nome}: f = {freqs[i]:.4f} Hz", fontsize=12, fontweight='bold')
    plt.xlabel("Direção (graus)"); plt.ylabel("D (1/grau)")
    plt.xlim(0, 360); plt.grid(alpha=0.5)
    plt.legend(title="RMSE em unidades de $10^{-3}$ grau$^{-1}$", fontsize=10)
    plt.savefig(arquivo, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"{nome}: f={freqs[i]:.4f} Hz | RMSE (1e-3 /grau): Fourier {rmse_f_g*1e3:.2f} | LK {rmse_lk_g*1e3:.2f} | HK {rmse_hk_g*1e3:.2f}")

grafico_cartesiano(i1,   "Frequência do pico do marulho", "sintetico_dois_sistemas_cartesiano_marulho.png")
grafico_cartesiano(i_ov, "Frequência de sobreposição",    "sintetico_dois_sistemas_cartesiano_sobreposicao.png")
grafico_cartesiano(i2,   "Frequência do pico da vaga",    "sintetico_dois_sistemas_cartesiano_vaga.png")