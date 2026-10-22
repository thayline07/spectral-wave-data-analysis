import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from wavespectra.input.ndbc_ascii import read_file, construct_spectra

ano = 2025
boia = "41004"
local = "Edisto, Carolina do Sul"

# ===================== CONFIGURACAO =====================
ARQUIVOS = {                      # os 5 arquivos da MESMA boia e do MESMO ano
    "swden":  f"dados/{boia}_{ano}/{boia}w{ano}.txt.gz",
    "swdir":  f"dados/{boia}_{ano}/{boia}d{ano}.txt.gz",    # alpha1
    "swdir2": f"dados/{boia}_{ano}/{boia}i{ano}.txt.gz",    # alpha2
    "swr1":   f"dados/{boia}_{ano}/{boia}j{ano}.txt.gz",    # r1
    "swr2":   f"dados/{boia}_{ano}/{boia}k{ano}.txt.gz",    # r2
}
INSTANTE = "2025-01-01 09:10"
SAIDA = "espectro_wavespectra.png"
# ========================================================

# Codigo de dado ausente nos arquivos historicos: 999 (99 e uma direcao
# valida nos arquivos de direcao, e 0,99 valido em r1 e r2).
FLAG = 999.0


def ler(caminho):
    df = read_file(caminho)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.where(df != FLAG)               # codigo de ausencia -> NaN


d = {nome: ler(caminho) for nome, caminho in ARQUIVOS.items()}

# O leitor do wavespectra NAO reescala r1 e r2. Nos arquivos historicos eles
# vem em centesimos, entao a escala e sempre 0.01 (como no download_arquivos.py).
escala_r = 0.01
r1_max = np.nanmax(d["swr1"].values) * escala_r
assert r1_max <= 1.0, f"r1 maximo apos a escala = {r1_max:.2f} (deveria ser <= 1)"
print(f"r1 maximo apos a escala = {r1_max:.2f}")

# Alinha os 5 arquivos nos mesmos horarios
tempos = d["swden"].index
for nome in d:
    d[nome] = d[nome].reindex(tempos)

i = tempos.get_indexer([np.datetime64(INSTANTE)], method="nearest")[0]
print(f"Instante pedido: {INSTANTE} | instante usado: {tempos[i]}")

forma = (1, d["swden"].shape[1], 1)           # (tempo, freq, dir)
pega = lambda nome, k=1.0: d[nome].values[i:i + 1].reshape(forma) * k
freqs = d["swden"].columns.astype(float).values
dirs = np.arange(0, 360, 1)                   # mesma malha de 1 grau do restante do trabalho

# Fourier truncado (2a ordem), sem ponderacao dos coeficientes
S = construct_spectra(pega("swden"), pega("swdir"), pega("swdir2"),
                      pega("swr1", escala_r), pega("swr2", escala_r),
                      dirs, weight_coeff=False)
efth = xr.DataArray(S[0], coords={"freq": freqs, "dir": dirs},
                    dims=("freq", "dir"), name="efth")

# vmin e vmax impressos em espectro_direcional_mem2.py para o mesmo instante e mesma boia
VMIN = -6.189645e-03   
VMAX = 1.850884e-01
niveis = np.linspace(VMIN, VMAX, 30)    

# niveis = np.linspace(float(efth.min()), float(efth.max()), 25)

efth.spec.plot(kind="contourf", normalised=False,
               logradius=False, rmax=0.35, cmap="turbo", figsize=(7, 6),
               levels=niveis)

plt.title("Fourier truncado (2ª ordem), wavespectra", fontsize=10)
plt.suptitle(
    f"$S(f,\\theta)$ em {tempos[i]:%d/%m/%Y %H:%M} UTC - Boia {boia} ({local})",
    y=1.02,
)
plt.savefig(SAIDA, dpi=150, bbox_inches="tight")
print(f"Figura salva em {SAIDA}")

w = d["swden"].values[i][:, None]
a1 = np.deg2rad(d["swdir"].values[i])[:, None]
a2 = np.deg2rad(d["swdir2"].values[i])[:, None]
r1 = (d["swr1"].values[i] * escala_r)[:, None]
r2 = (d["swr2"].values[i] * escala_r)[:, None]
th = np.deg2rad(dirs)[None, :]

D = (0.5 + r1 * np.cos(th - a1) + r2 * np.cos(2 * (th - a2))) / np.pi   # por radiano
S_propria = w * D * np.pi / 180                                         # por grau

dif = np.nanmax(np.abs(S_propria - efth.values))
pico = np.nanmax(efth.values)
print(f"Fourier truncado: maior diferenca absoluta = {dif:.2e} m2/Hz/grau "
      f"(pico do espectro = {pico:.2e})")