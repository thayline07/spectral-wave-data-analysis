# Processamento Espectral e Análise Estatística de Dados de Ondas

Repositório dedicado ao desenvolvimento de rotinas em Python para automação de download, controle de qualidade, análise e reconstrução espectral direcional de dados de ondas, utilizando como base histórica as estações oceanográficas do **National Data Buoy Center (NDBC/NOAA)**. A distribuição direcional da energia é reconstruída por três métodos: Fourier truncado, Máxima Entropia (MEM) de Lygre e Krogstad e MEM de Hashimoto e Kobune. O MEM de Hashimoto e Kobune é o método adotado para as análises; Fourier e Lygre e Krogstad ficam como comparação.

---

## Boias e anos analisados

| Boia | Localidade | Anos |
|------|------------|------|
| 41004 | Edisto, Carolina do Sul | 2023, 2024, 2025 |
| 41040 | Atlântico Norte, a leste das Pequenas Antilhas | 2019, 2020, 2021 |
| 46080 | Portlock Bank, Golfo do Alasca | 2023, 2024, 2025 |

A 46080 apresenta lacunas extensas em 2023 e 2024 (a auditoria de `analise.py` quantifica a perda de registros de cada série).

---

## Estrutura do Projeto e Scripts

O projeto foi modularizado em scripts independentes e especializados para garantir organização, reaproveitamento de dados e facilidade de manutenção:

1. **`download_arquivos.py`**
   - **Função**: Acessa automaticamente via internet o servidor histórico do NDBC/NOAA. Realiza o download dos arquivos compactados (`.txt.gz`) dos 5 parâmetros essenciais: densidade de energia (`w`), direções de propagação (`alpha1`, `alpha2`) e coeficientes de espalhamento de Fourier (`r1`, `r2`). Arquivos já existentes na pasta local não são baixados novamente.
   - **Controle de Qualidade (QC)**: Nos arquivos históricos, dado ausente é indicado por uma sequência de noves: `999.0` nos arquivos de direção (`alpha1`, `alpha2`) e `999.00` nos de `r1` e `r2`. Esses códigos são substituídos por `NaN`. Em seguida, os coeficientes $r_1$ e $r_2$ válidos têm a escala física corrigida, multiplicando-os por $0.01$ (os arquivos originais armazenam esses valores em centésimos). Nos arquivos de direção, `99` **não** é código de ausência: é uma direção válida (99°). Ao final, consolida tudo em um único dicionário binário compactado (`.pkl`).
   - **Uso como módulo**: Toda a rotina está encapsulada na função `baixar_dados(id_boia, year)`, que retorna o caminho do `.pkl` gerado ou `None` caso algum dos 5 arquivos não exista no servidor. Isso permite que outros scripts baixem automaticamente anos que ainda não estejam disponíveis localmente.

2. **`wind.py`**
   - **Função**: Realiza o download dos dados meteorológicos padrão da boia (pasta `stdmet` do NDBC, arquivos com sufixo `h`), contendo a direção (`WDIR`) e a velocidade (`WSPD`) do vento local e a altura significativa de onda publicada pelo NDBC (`WVHT`). Se o download não for concluído ou o arquivo não existir, o script informa o problema e encerra.
   - **Controle de Qualidade (QC)**: Substitui os códigos de ausência de medição (`999` para direção, `99.0` para velocidade e `99.0` para `WVHT`) por valores nulos (`NaN`).
   - **Saída**: Cria o índice cronológico `data_registro` e exporta as séries em formato `.csv` (`wind_{boia}_{ano}.csv`), com as colunas `dir` (vento), `vel` (vento) e `height` (`WVHT`). O `wind.py` deve ser executado **antes** do `analise.py`, que usa a coluna `height`, e é também a fonte do vento usado nos scripts de reconstrução direcional.

3. **`analise.py`**
   - **Função**: Carrega os dados limpos do arquivo `.pkl` e realiza o cálculo dos parâmetros integrados registro a registro para o ano inteiro.
   - **Cálculo de $H_s$**: Obtém a Altura Significativa calculando o momento de ordem zero ($m_0$), que representa a área total sob a curva do espectro de energia ($H_s = 4\sqrt{m_0}$), utilizando a largura de cada faixa de frequência (`np.diff`; a última faixa repete a largura da anterior). O NDBC calcula o `WVHT` também como $4\sqrt{m_0}$, mas com a largura de banda informada para cada sistema de medição (de 0.005 Hz a 0.02 Hz), o que torna a comparação com o `WVHT` uma validação independente.
   - **Cálculo de $T_p$**: Localiza a frequência de pico onde ocorre a maior densidade energética e extrai o Período de Pico através do seu inverso ($T_p = 1/f_p$).
   - **Auditoria Temporal**: Remove registros duplicados e reindexa a série sobre um calendário completo no intervalo de amostragem da boia, expondo os horários sem registro como `NaN` para a contagem de falhas (*gaps*).
   - **Cruzamento com o `WVHT`**: Se existir o `wind_{boia}_{ano}.csv`, cada registro espectral recebe o `WVHT` mais próximo no tempo (`pd.merge_asof`, direção `nearest`, tolerância de 20 minutos), na coluna `height`. Se o arquivo não existir, o script avisa e gera a série sem essa coluna.
   - **Saída**: Cria um índice cronológico unificado (`data_registro`) e exporta as séries temporais consolidadas em formato `.csv` (`series_temporais_{boia}_{ano}.csv`), com `Hs`, `Tp` e `height`.

4. **`histogramas.py`**
   - **Função**: Lê as séries temporais em `.csv` de todos os anos de cada boia, concatena e remove duplicatas, e gera as distribuições de frequência estatística das variáveis marinhas para todo o período, sem separar por ano.
   - **Análise Física**: Produz gráficos de barras independentes para identificar o clima de ondas padrão da estação, assimetrias temporais, limites de normalidade (faixas de 2.5m a 3.0m) e dominâncias dinâmicas de sistemas de vagas (*wind sea* de períodos curtos) e marulhos (*swell* de longos períodos). Salvando os outputs automaticamente em imagens `.png` (`histograma_hs_{boia}.png` e `histograma_tp_{boia}.png`). Os títulos indicam a boia, a localidade e o período. O histograma de $H_s$ usa 25 classes e o de $T_p$, 15 classes.
   - **Estatísticas de $H_s$**: Marca no histograma a mediana, o valor máximo e o percentil 99 ($P_{99}$, valor de $H_s$ ultrapassado em apenas 1% dos registros), que descreve o mar de tempestade do local de forma mais robusta que o máximo, por não depender de um único evento.
   - **Escala do Eixo Vertical**: Utiliza escala de raiz quadrada no eixo das contagens, comprimindo as classes com milhares de registros mais do que as classes com poucas dezenas. Isso torna visível a cauda da distribuição (eventos extremos), mantendo no eixo os valores reais de contagem. O eixo vertical do histograma de $H_s$ indica essa transformação ("escala de raiz quadrada") e o do histograma de $T_p$ indica "número de registros".
   - **Verificação de Extremos**: A função `checar_vizinhos` localiza o registro de $H_s$ máximo e exibe os registros vizinhos, permitindo verificar se o valor pertence a um evento contínuo (tempestade) ou se é um valor isolado e possivelmente espúrio.

5. **`valida_hs.py`**
   - **Função**: Valida o cálculo de $H_s$ do `analise.py` contra o `WVHT` publicado pelo próprio NDBC (coluna `height` das séries temporais). Para cada boia, usa os registros pareados de todos os anos.
   - **Saída**: Um gráfico de dispersão por boia (`validacao_hs_{boia}.png`), com $H_s$ calculado no eixo horizontal, `WVHT` no eixo vertical e a reta $y = x$. Os pontos são coloridos pela densidade de registros (escala logarítmica). Cada figura indica o número de registros pareados ($N$), o erro médio (média de $H_s - \mathrm{WVHT}$), o RMSE e o coeficiente de correlação de Pearson ($r$).
   - **Requisito**: precisa dos `series_temporais_{boia}_{ano}.csv` com a coluna `height`, gerados pelo `analise.py` depois do `wind.py`.

6. **`espectro-teste.py`**
   - **Função**: Laboratório numérico de validação controlada. Cria um espectro direcional artificial utilizando uma distribuição Gaussiana na frequência e um espalhamento cossenoidal focado em uma direção (90° Leste).
   - **Objetivo**: Aplica a engenharia reversa para extrair os coeficientes e testa o comportamento intrínseco dos algoritmos frente a dados perfeitos, permitindo diagnosticar limitações teóricas como o fenômeno de *peak-splitting* e vazamento de energia.

7. **`espectro_direcional.py`**
   - **Função**: Script de reconstrução bidimensional $S(f, \theta)$ para um instante específico (data e hora definidas em `data_alvo`), utilizando dados reais medidos pelas boias do NDBC. Os espectros são registrados nos minutos 10 e 40 de cada hora.
   - **Modelagem**: Processa as matrizes de energia para cada uma das 47 frequências em todo o círculo trigonométrico (malha de 1°, de 0° a 360°; o ponto de 360° repete o de 0° apenas para fechar o gráfico polar) aplicando simultaneamente três abordagens: a **Expansão Linear de Fourier** (2ª ordem truncada), o **MEM de Lygre e Krogstad** (entropia de Burg) e o **MEM de Hashimoto e Kobune** (entropia de Shannon, resolvida por Newton com busca de passo de Armijo). Frequências cujos momentos não são realizáveis (condição de viabilidade na seção de fundamentação teórica) ficam como `NaN` nos MEM e aparecem em branco.
   - **Integração com o Vento**: Como os registros de vento e de espectro não ocorrem nos mesmos minutos, o vento correspondente é localizado por proximidade temporal com `pd.merge_asof` (direção `nearest`, tolerância de 20 minutos). Caso não exista vento válido dentro da tolerância, o script informa a ausência.
   - **Output**: Renderiza mapas de calor coloridos (`contourf` polar em coordenadas geográficas/náuticas) com os três métodos em painéis lado a lado, usando os mesmos níveis de cores (do menor ao maior valor entre os três) e uma única barra de cores, com a direção e a velocidade do vento indicadas na figura. Os títulos incluem a boia e a localidade. Gera também um gráfico cartesiano de $D(\theta)$ na frequência de pico, sobrepondo os três métodos e uma linha vertical na direção do vento. O eixo vertical está em 1/grau.

8. **`espectro_direcional_data_especifica.py`**
   - **Função**: Reconstrução bidimensional $S(f, \theta)$ da média diária de um dia escolhido (`dia_alvo` e `mes_alvo`), aplicando os três métodos (Fourier, MEM de Lygre e Krogstad e MEM de Hashimoto e Kobune).
   - **Média dos Momentos**: A média é calculada sobre os momentos complexos de cada registro ($c_1 = r_1e^{i\alpha_1}$ e $c_2 = r_2e^{2i\alpha_2}$), ponderados pela densidade de energia `w`. Registros com valores nulos recebem peso zero. Os detalhes estão na seção de fundamentação teórica.
   - **Verificação**: Exibe a integral de $D(\theta)$ em cada frequência pela regra dos trapézios (`np.trapezoid`), que deve resultar em 1 para os métodos.
   - **Output**: Gera os mapas polares da média diária e o gráfico cartesiano de $D(\theta)$ na frequência de pico, com a boia e a localidade nos títulos. Nestes dois gráficos, $D(\theta)$ e a densidade de energia estão em radianos (1/rad e m²/Hz/rad), enquanto os gráficos de instante específico usam graus.

9. **`espectro_direcional_estacoes.py`**
   - **Função**: Reconstrução bidimensional $S(f, \theta)$ média para cada estação do ano do Hemisfério Norte (Inverno, Primavera, Verão e Outono), utilizando a mesma média ponderada dos momentos complexos do script diário e os três métodos.
   - **Inverno**: O inverno de um ano $Y$ é composto por dezembro de $Y-1$, janeiro e fevereiro de $Y$. Dezembro de $Y$ pertence ao inverno seguinte e não é incluído. Caso os dados de $Y-1$ não estejam disponíveis localmente, o script tenta baixá-los com `baixar_dados`. Se o ano anterior não existir no servidor, o inverno é calculado apenas com janeiro e fevereiro, e o título da figura indica essa limitação.
   - **Output**: Gera uma figura com os três mapas polares (Fourier, MEM de Lygre e Krogstad e MEM de Hashimoto e Kobune) para cada estação, com a boia, a localidade e o período utilizado (meses e ano) no título, por exemplo "Primavera mar a mai/2023". As unidades são as mesmas do script diário (rad).

10. **`espectro_wavespectra.py`**
    - **Função**: Verifica a reconstrução de Fourier contra a biblioteca [`wavespectra`](https://github.com/wavespectra/wavespectra), que lê os mesmos cinco arquivos `.txt.gz` originais com `read_file` e reconstrói $S(f,\theta)$ com `construct_spectra` (Fourier truncado na 2ª ordem, malha de 1°). A biblioteca não implementa MEM.
    - **Tratamento dos dados**: remove duplicatas, troca o código de ausência `999` por `NaN`, alinha os cinco arquivos nos mesmos horários e aplica o fator $0.01$ a $r_1$ e $r_2$, conferindo que o máximo de $r_1$ não ultrapassa 1 (os arquivos de direção não são alterados em `99`).
    - **Verificação numérica**: calcula a maior diferença absoluta entre o espectro obtido pela rotina própria de Fourier e o da biblioteca. Para a boia 41004 em 01/01/2025 às 09:10 UTC, a diferença máxima foi de $2.43\times10^{-17}$ m²/Hz/grau, frente a um pico de $3.94\times10^{-2}$ m²/Hz/grau, isto é, da ordem do erro de arredondamento.
    - **Output**: mapa polar `espectro_wavespectra.png`, que pode usar os mesmos níveis de cores do mapa de Fourier da reconstrução direcional do mesmo instante.

---

## Fundamentação Teórica dos Métodos Reconstruídos

Em cada frequência, as boias fornecem apenas os dois primeiros coeficientes de Fourier da distribuição direcional $D(\theta)$, com $\int_0^{2\pi} D(\theta)\,d\theta = 1$:

$$c_1 = \int_0^{2\pi} D(\theta)\,e^{i\theta}\,d\theta = r_1 e^{i\alpha_1}, \qquad c_2 = \int_0^{2\pi} D(\theta)\,e^{2i\theta}\,d\theta = r_2 e^{2i\alpha_2}$$

Qualquer $D$ que satisfaça essas igualdades é compatível com os dados, e cada método abaixo escolhe uma solução diferente. As direções seguem a convenção do NDBC: de onde a onda vem, medida em sentido horário a partir do Norte.

### Expansão da Série de Fourier (Truncada)
Assume que o espalhamento direcional pode ser aproximado por uma combinação linear suave de harmônicos senoidais. Devido às limitações físicas de medição das boias triaxiais, a série é cortada (**truncada**) na segunda ordem:
$$D_{\text{fourier}}(\theta) = \frac{1}{\pi} \left[ \frac{1}{2} + r_1\cos(\theta - \alpha_1) + r_2\cos(2(\theta - \alpha_2)) \right]$$
* **Limitações**: Apresenta baixa resolução angular (gerando borrões largos ao redor do pico) e sofre com oscilações numéricas que produzem **energias negativas artificiais**, violando leis físicas básicas. Por ser linear, sua integral é sempre igual a 1, mesmo quando os momentos de entrada são inconsistentes, de modo que o método não sinaliza esse tipo de problema.

### Método da Máxima Entropia (MEM) de Lygre e Krogstad
Abordagem estatística baseada na entropia de Burg. Busca a distribuição direcional mais aleatória e menos tendenciosa possível para os componentes desconhecidos, vinculada estritamente aos 4 momentos medidos pela boia. É calculada de forma complexa resolvendo as equações de Yule-Walker:
$$D_{\text{MEM}}(\theta) = \frac{1}{2\pi} \frac{1 - \phi_1 c_1^* - \phi_2 c_2^*}{|1 - \phi_1 e^{-i\theta} - \phi_2 e^{-i2\theta}|^2}$$
com
$$\phi_2 = \frac{c_2 - c_1^2}{1 - |c_1|^2}, \qquad \phi_1 = c_1 - \phi_2\,c_1^*$$
* **Vantagens**: Proporciona altíssima resolução angular (picos extremamente estreitos e focados) e reproduz exatamente os momentos medidos. O denominador, por ser um módulo elevado ao quadrado, nunca é negativo.
* **Limitações**: O sinal de $D_{\text{MEM}}$ é determinado pelo numerador. Quando os momentos não são fisicamente realizáveis, isto é, quando nenhuma distribuição direcional real poderia produzi-los ($|c_2 - c_1^2| > 1 - |c_1|^2$), o numerador torna-se negativo e a curva inteira é invertida, com integral igual a $-1$. Além disso, para espalhamentos unimodais com $r_2$ baixo em relação a $r_1$, o método pode dividir um único pico em dois (*peak-splitting*), de modo que um pico duplo no MEM não comprova, isoladamente, a existência de um mar bimodal. Esse efeito foi observado em 31–46% dos registros, na frequência de pico, dos casos testados (41004/2023, 46080/2023 e 41040/2020), e também no espectro sintético do `espectro-teste.py`.

### Método da Máxima Entropia de Hashimoto e Kobune
Usa o mesmo princípio, mas maximiza a entropia de Shannon, $H_S = -\int D\ln D\,d\theta$, sujeita aos mesmos 4 momentos. A solução é uma distribuição exponencial, sempre positiva e suave:
$$D(\theta) = \frac{1}{Z(\boldsymbol{\lambda})}\exp\!\big[-\boldsymbol{\lambda}\cdot\boldsymbol{\varphi}(\theta)\big], \qquad \boldsymbol{\varphi} = (\cos\theta,\ \sin\theta,\ \cos 2\theta,\ \sin 2\theta)$$
em que $Z(\boldsymbol{\lambda}) = \int_0^{2\pi}\exp[-\boldsymbol{\lambda}\cdot\boldsymbol{\varphi}]\,d\theta$ normaliza a distribuição. Não há fórmula fechada para os quatro multiplicadores $\boldsymbol{\lambda}$: eles são obtidos resolvendo o problema dual, convexo,
$$\min_{\boldsymbol{\lambda}}\ F(\boldsymbol{\lambda}) = \ln Z(\boldsymbol{\lambda}) + \boldsymbol{\lambda}\cdot\mathbf{m}, \qquad \mathbf{m} = \big(\operatorname{Re}c_1,\ \operatorname{Im}c_1,\ \operatorname{Re}c_2,\ \operatorname{Im}c_2\big)$$
* **Algoritmo** (função `mem_shannon`): parte de $\boldsymbol{\lambda}=\mathbf{0}$ (distribuição uniforme), calcula o gradiente $\mathbf{m} - E[\boldsymbol{\varphi}]$ e o hessiano (matriz de covariância de $\boldsymbol{\varphi}$), dá o passo de Newton com busca de passo de Armijo (reduz o passo pela metade até $F$ diminuir o suficiente) e para quando $\max|\mathbf{m}-E[\boldsymbol{\varphi}]| < 10^{-10}$, com no máximo 200 iterações. O máximo do expoente é subtraído antes de exponenciar, para evitar *overflow*.
* **Método adotado**: é o método adotado para as análises deste projeto. Fourier e MEM de Lygre e Krogstad são mantidos como comparação.

### Viabilidade dos Momentos
Nem todo par $(c_1, c_2)$ medido corresponde a uma distribuição não negativa. Isso só ocorre quando
$$|c_1| < 1 \quad \text{e} \quad |c_2 - c_1^2| < 1 - |c_1|^2$$
Quando a condição não vale (por ruído do instrumento ou arredondamento dos arquivos), o método de Lygre e Krogstad devolve densidades negativas sem avisar e o de Hashimoto e Kobune não converge. Nesses casos a frequência é marcada como `NaN` nos dois MEM, aparece em branco nos gráficos e é listada no terminal.

### Média Temporal dos Parâmetros Direcionais
Os parâmetros $r$ e $\alpha$ não são independentes: juntos formam os momentos complexos $c_1 = r_1e^{i\alpha_1}$ e $c_2 = r_2e^{2i\alpha_2}$. Calcular a média de $r$ e de $\alpha$ separadamente superestima a concentração direcional quando a direção varia ao longo do período, o que pode gerar momentos não realizáveis. Por isso, as médias diária e sazonal são calculadas sobre os momentos complexos, ponderados pela densidade de energia:
$$\bar{c}_n = \frac{\sum w \, c_n}{\sum w}$$
Esse procedimento equivale a calcular a média do espectro direcional $S(f, \theta)$ e garante que os momentos médios sejam sempre realizáveis. Depois, $r_1 = |\bar{c}_1|$, $\alpha_1 = \arg \bar{c}_1$, $r_2 = |\bar{c}_2|$ e $\alpha_2 = \tfrac{1}{2}\arg \bar{c}_2$.

Uma média diária ou sazonal soma espectros de instantes diferentes: dois picos na média podem vir de sistemas que ocorreram em dias distintos, sem nunca terem coexistido. A coexistência de sistemas de ondas (por exemplo, vaga e marulho) só pode ser identificada no espectro de um único instante.

### Validação de $H_s$ contra o `WVHT` (`valida_hs.py`)
O erro médio, o RMSE e o coeficiente de correlação de Pearson são calculados sobre os $N$ registros pareados:
$$\text{erro médio} = \frac{1}{N}\sum_{i=1}^{N}\left(H_{s,i} - \mathrm{WVHT}_i\right), \qquad \mathrm{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(H_{s,i} - \mathrm{WVHT}_i\right)^2}$$

---

## Como Executar as Rotinas

Os códigos estão preparados para operar sequencialmente de forma automatizada. Garanta que o ambiente possua as bibliotecas `pandas`, `numpy` e `matplotlib` devidamente instaladas (para o `espectro_wavespectra.py`, também `xarray` e `wavespectra`). Em cada script, o ID da boia e o ano são definidos no cabeçalho (`id_boia` e `year`). Todos os dados são salvos e lidos na pasta `dados/{boia}_{ano}/`, relativa à pasta de onde os scripts são executados:

1. Altere o ID da boia e o ano desejado no cabeçalho do script de automação e execute o download/limpeza:
   ```bash
   python download_arquivos.py
   ```
2. Baixe e trate os dados de vento e o `WVHT` da mesma boia e ano (necessário antes do `analise.py`):
   ```bash
   python wind.py
   ```
3. Processe a matemática das séries temporais integradas de altura e período:
   ```bash
   python analise.py
   ```
4. Gere as análises gráficas estatísticas de histogramas (todos os anos da boia):
   ```bash
   python histogramas.py
   ```
5. Compare o $H_s$ calculado com o `WVHT` do NDBC (um gráfico por boia):
   ```bash
   python valida_hs.py
   ```
6. Para simular e validar os códigos no ambiente controlado do laboratório sintético:
   ```bash
   python espectro-teste.py
   ```
7. Para renderizar os mapas de calor polares e comparar os três métodos direcionais com dados reais do oceano em um instante específico:
   ```bash
   python espectro_direcional.py
   ```
8. Para a reconstrução direcional média de um dia específico:
   ```bash
   python espectro_direcional_data_especifica.py
   ```
9. Para a reconstrução direcional média de cada estação do ano:
   ```bash
   python espectro_direcional_estacoes.py
   ```
10. Para conferir a reconstrução de Fourier com a biblioteca `wavespectra`:
    ```bash
    python espectro_wavespectra.py
    ```
