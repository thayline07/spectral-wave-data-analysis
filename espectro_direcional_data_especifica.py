import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Boias escolhidas:
# 41004 (2023, 2024, 2025)
# 41040 (2019, 2020, 2021)
# 46080 (2023, 2024, 2025)

id_boia = "46080"
year = 2023

locais = {"41004": "Edisto, Carolina do Sul",
          "41040": "Atlântico Norte, a leste das Pequenas Antilhas",
          "46080": "Portlock Bank, Golfo do Alasca"}

caminho_arq = os.path.join(
    "dados", f"{id_boia}_{year}", f"dados_{id_boia}_{year}.pkl"
)

if os.path.exists(caminho_arq):
    with open(caminho_arq, "rb") as arq:
        dados = pickle.load(arq)
    print(f"Arquivo '{caminho_arq}' carregado com sucesso!\n")
else:
    print(
        f" Arquivo '{caminho_arq}' não encontrado. Por favor, execute o download primeiro."
    )
    exit(1)

df_w = dados["w"]
df_alpha1 = dados["alpha1"]
df_alpha2 = dados["alpha2"]
df_r1 = dados["r1"]
df_r2 = dados["r2"]

frequencias = np.array([float(col) for col in df_w.columns[5:]])

# Criação da coluna de data para filtro temporizado
df_w["data_registro"] = pd.to_datetime(
    pd.DataFrame({
        "year": df_w["#YY"],
        "month": df_w["MM"],
        "day": df_w["DD"],
        "hour": df_w["hh"],
        "minute": df_w["mm"]
    })
)


# resultados críticos: 02/12, 
dia_alvo = 2
mes_alvo = 6

# Filtra os índices que pertencem ao dia escolhido
indices_dia = df_w[
    (df_w["data_registro"].dt.day == dia_alvo)
    & (df_w["data_registro"].dt.month == mes_alvo)
].index

if len(indices_dia) == 0:
    print(
        f"Nenhum dado encontrado para o dia {dia_alvo:02d}/{mes_alvo:02d}/{year}."
    )
    exit(1)

w_dia = df_w.iloc[indices_dia, 5:-1].values.astype(float)
r1_dia = df_r1.iloc[indices_dia, 5:].values.astype(float)
r2_dia = df_r2.iloc[indices_dia, 5:].values.astype(float)
alpha1_dia = np.radians(df_alpha1.iloc[indices_dia, 5:].values.astype(float))
alpha2_dia = np.radians(df_alpha2.iloc[indices_dia, 5:].values.astype(float))

c1_dia = r1_dia * np.exp(1j * alpha1_dia)
c2_dia = r2_dia * np.exp(2j * alpha2_dia)

validos = ~(np.isnan(w_dia) | np.isnan(c1_dia) | np.isnan(c2_dia))
peso = np.where(validos, w_dia, 0.0)
soma_pesos = peso.sum(axis=0)

with np.errstate(divide="ignore", invalid="ignore"):
    c1_medio = np.sum(peso * np.where(validos, c1_dia, 0), axis=0) / soma_pesos
    c2_medio = np.sum(peso * np.where(validos, c2_dia, 0), axis=0) / soma_pesos

w_medio = np.nanmean(w_dia, axis=0)

angulos_graus = np.arange(0, 361, 1)
angulos_rad = np.radians(angulos_graus)

matriz_fourier = np.zeros((len(frequencias), len(angulos_graus)))
matriz_max_entropia = np.zeros((len(frequencias), len(angulos_graus)))
matriz_shannon = np.zeros((len(frequencias), len(angulos_graus)))
sem_convergir = []


# ---------------- MEM de Kobune e Hashimoto (entropia de Shannon) ----------------
# Passo da malha e vetor das quatro funções-base (cos t, sen t, cos 2t, sen 2t).
dth = angulos_rad[1] - angulos_rad[0]
phi = np.array([np.cos(angulos_rad), np.sin(angulos_rad), np.cos(2 * angulos_rad), np.sin(2 * angulos_rad)])


def _dist(lam):
    # D(theta) = exp(-lam . phi) / Z, com o máximo subtraído para evitar overflow
    logw = -(lam @ phi)
    mx = logw.max()
    ew = np.exp(logw - mx)
    Z = ew.sum() * dth
    return ew / Z, np.log(Z) + mx


def mem_shannon(m, tol=1e-10, maxiter=200):
    # Acha os multiplicadores lam por Newton com busca de passo (problema dual, convexo).
    # m = (Re c1, Im c1, Re c2, Im c2). Devolve (D, convergiu).
    lam = np.zeros(4)
    D, logZ = _dist(lam)
    F = logZ + lam @ m

    for _ in range(maxiter):
        E = phi @ D * dth
        grad = m - E
        if np.max(np.abs(grad)) < tol:
            return D, True

        cov = (phi * D) @ phi.T * dth - np.outer(E, E)
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


for freq in range(len(frequencias)):
    w_f = w_medio[freq]
    c1 = c1_medio[freq]
    c2 = c2_medio[freq]

    r1_f = np.abs(c1)
    a1_f = np.angle(c1)
    r2_f = np.abs(c2)
    a2_f = np.angle(c2)/2.0

    if np.isnan([w_f, r1_f, r2_f, a1_f, a2_f]).any():
        continue

    # Fourier
    D_fourier = (1.0 / np.pi) * (
        0.5
        + (r1_f * np.cos(angulos_rad - a1_f))
        + (r2_f * np.cos(2.0 * (angulos_rad - a2_f)))
    )
    matriz_fourier[freq, :] = D_fourier * w_f


    # Os momentos só correspondem a uma D(theta) não negativa se esta condição vale.
    viavel = (r1_f < 1.0) and (np.abs(c2 - c1**2) < 1.0 - r1_f**2)
    if not viavel:
        print(f"{frequencias[freq]:.3f} Hz: momentos fora da região viável, MEM indefinido. Pulando.")
        matriz_max_entropia[freq, :] = np.nan
        matriz_shannon[freq, :] = np.nan
        sem_convergir.append(frequencias[freq])
        continue

    # Máxima entropia (Lygre-Krogstad)

    phi2 = (c2 - (c1**2)) / (1.0 - (np.abs(c1) ** 2))
    phi1 = c1 - (phi2 * np.conj(c1))

    termo_exp1 = np.exp(-1j * angulos_rad)
    termo_exp2 = np.exp(-2j * angulos_rad)

    numerador = 1.0 - np.real(phi1 * np.conj(c1) + phi2 * np.conj(c2))
    denominador = (
        np.abs(1.0 - (phi1 * termo_exp1) - (phi2 * termo_exp2)) ** 2
    )
    

    D_max_entropia = (1.0 / (2.0 * np.pi)) * (numerador / denominador)
    matriz_max_entropia[freq, :] = D_max_entropia.real * w_f

    # Máxima entropia (Kobune e Hashimoto)
    m = np.array([c1.real, c1.imag, c2.real, c2.imag])
    D_shannon, ok = mem_shannon(m)
    if ok:
        matriz_shannon[freq, :] = D_shannon * w_f
    else:
        matriz_shannon[freq, :] = np.nan
        sem_convergir.append(frequencias[freq])

    print("Area Fourier:", np.trapezoid(D_fourier, angulos_rad), ", Area MEM:", np.trapezoid(D_max_entropia.real, angulos_rad), ", Area Shannon:", np.trapezoid(D_shannon, angulos_rad))

print(f"Frequências sem MEM: {sem_convergir}")

Angulos, Freqs = np.meshgrid(angulos_rad, frequencias)

todas = np.concatenate([matriz_fourier.ravel(), matriz_max_entropia.ravel(), matriz_shannon.ravel()])
v_min, v_max = np.nanmin(todas), np.nanmax(todas)
niveis_cores = np.linspace(v_min, v_max, 30)

fig1, axes1 = plt.subplots(
    1, 3, figsize=(20, 6), subplot_kw={"projection": "polar"}, layout="constrained"
)
fig1.suptitle(
    f"Boia {id_boia} ({locais[id_boia]}) - Média diária de {dia_alvo:02d}/{mes_alvo:02d}/{year}",
    fontsize=15,
    fontweight="bold",
)

contorno1 = axes1[0].contourf(
    Angulos, Freqs, matriz_fourier, cmap="jet", levels=niveis_cores
)
axes1[0].set_title("Reconstrução por Fourier", fontsize=12, fontweight="bold")

contorno2 = axes1[1].contourf(
    Angulos, Freqs, matriz_max_entropia, cmap="jet", levels=niveis_cores
)
axes1[1].set_title(
    "Reconstrução MEM\n(Lygre-Krogstad)", fontsize=12, fontweight="bold"
)

contorno3 = axes1[2].contourf(
    Angulos, Freqs, matriz_shannon, cmap="jet", levels=niveis_cores
)
axes1[2].set_title(
    "Reconstrução MEM\n(Kobune e Hashimoto)", fontsize=12, fontweight="bold"
)

for ax in axes1:
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim([frequencias.min(), frequencias.max()])

fig1.colorbar(
    contorno3, ax=axes1, shrink=0.75, pad=0.04, label="Densidade de Energia ($m^2/Hz/rad$)"
)

caminho_polar = f"reconstrucao_polar_media_{id_boia}_{year}.png"
plt.savefig(caminho_polar, dpi=300, bbox_inches="tight")
print(f"Figura 1 (Polares) salva em: {caminho_polar}")
plt.show()

#Gráfico cartesiano
f_pico_idx = np.argmax(w_medio)
f_pico_valor = frequencias[f_pico_idx]

plt.figure(figsize=(9, 5))

w_pico = w_medio[f_pico_idx]
D_fourier_pico = matriz_fourier[f_pico_idx, :] / w_pico
D_mem_pico = matriz_max_entropia[f_pico_idx, :] / w_pico
D_shannon_pico = matriz_shannon[f_pico_idx, :] / w_pico

plt.plot(
    angulos_graus,
    D_fourier_pico,
    label="Expansão de Fourier",
    color="orange",
    linewidth=2.5,
)
plt.plot(
    angulos_graus,
    D_mem_pico,
    label="MEM (Lygre-Krogstad)",
    color="green",
    linestyle="-.",
    linewidth=2.5,
)
plt.plot(
    angulos_graus,
    D_shannon_pico,
    label="MEM (Kobune-Hashimoto)",
    color="purple",
    linewidth=2.5,
)

plt.title(
    f"Boia {id_boia} ({locais[id_boia]})\nComparação Direta de $D(\\theta)$ na Frequência de Pico ({f_pico_valor:.3f} Hz)\nMédia Diária - Data: {dia_alvo:02d}/{mes_alvo:02d}/{year}",
    fontsize=12,
    fontweight="bold",
    pad=12,
)
plt.xlabel("Direção Angular de Incidência (Graus)", fontsize=11)
plt.ylabel("Densidade Direcional $D(\\theta)$ ($rad^{-1}$)", fontsize=11)
plt.xlim(0, 360)
plt.xticks(np.arange(0, 361, 45))
plt.grid(linestyle="--", alpha=0.5)
plt.legend(fontsize=10, loc="upper right")

plt.tight_layout()
caminho_cartesiano = f"comparacao_cartesiana_media_{id_boia}_{year}.png"
#plt.savefig(caminho_cartesiano, dpi=300)
print(f" Figura 2 (Gráfico Cartesiano) salva em: {caminho_cartesiano}")
plt.show()