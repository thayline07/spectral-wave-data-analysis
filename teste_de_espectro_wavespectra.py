import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from wavespectra.input.ndbc_ascii import read_file, construct_spectra

ano = 2025
boia = "46080"

# ===================== CONFIGURACAO =====================
ARQUIVOS = {                      # os 5 arquivos da MESMA boia e do MESMO ano
    "swden":  f"dados/{boia}_{ano}/{boia}w{ano}.txt.gz",
    "swdir":  f"dados/{boia}_{ano}/{boia}d{ano}.txt.gz",    # alpha1
    "swdir2": f"dados/{boia}_{ano}/{boia}i{ano}.txt.gz",    # alpha2
    "swr1":   f"dados/{boia}_{ano}/{boia}j{ano}.txt.gz",    # r1
    "swr2":   f"dados/{boia}_{ano}/{boia}k{ano}.txt.gz",    # r2
}
INSTANTE = "2025-01-01 09:40"     # o mais proximo desse horario sera usado
SAIDA = "espectro_wavespectra.png"
# ========================================================

FLAGS = (99.0, 999.0)


def ler(caminho):
    df = read_file(caminho)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.where(~df.isin(FLAGS))          # flags de erro -> NaN


d = {nome: ler(caminho) for nome, caminho in ARQUIVOS.items()}

# O leitor do wavespectra NAO reescala r1 e r2. Se os seus valores vierem em
# centesimos (maiores que 1), aplica o 0.01 que voce ja usa no analise.py.
r1_max = np.nanmax(d["swr1"].values)
escala_r = 0.01 if r1_max > 1.0 else 1.0
print(f"r1 maximo no arquivo = {r1_max:.2f} -> escala aplicada em r1 e r2: {escala_r}")

# Alinha os 5 arquivos nos mesmos horarios
tempos = d["swden"].index
for nome in d:
    d[nome] = d[nome].reindex(tempos)

i = tempos.get_indexer([np.datetime64(INSTANTE)], method="nearest")[0]
print(f"Instante usado: {tempos[i]}")

forma = (1, d["swden"].shape[1], 1)           # (tempo, freq, dir)
pega = lambda nome, k=1.0: d[nome].values[i:i + 1].reshape(forma) * k
freqs = d["swden"].columns.astype(float).values
dirs = np.arange(0, 360, 5)

paineis = []
for rotulo, peso in [("Fourier (truncada em 2a ordem)", False),
                     ("Fourier com pesos (sem valores negativos)", True)]:
    S = construct_spectra(pega("swden"), pega("swdir"), pega("swdir2"),
                          pega("swr1", escala_r), pega("swr2", escala_r),
                          dirs, weight_coeff=peso)
    paineis.append(xr.DataArray(S[0], coords={"freq": freqs, "dir": dirs},
                                dims=("freq", "dir"), name="efth"))

efth = xr.concat(paineis, dim="metodo")
efth["metodo"] = ["Fourier", "Fourier com pesos"]

efth.spec.plot(col="metodo", kind="contourf", normalised=False,
               logradius=False, rmax=0.35, cmap="turbo", figsize=(11, 5.5))
plt.suptitle(f"S(f, theta) em {tempos[i]}", y=1.02)
plt.savefig(SAIDA, dpi=150, bbox_inches="tight")
print(f"Figura salva em {SAIDA}")
