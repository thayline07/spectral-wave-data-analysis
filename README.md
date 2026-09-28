# Processamento Espectral e Análise Estatística de Dados de Ondas

Repositório dedicado ao desenvolvimento de rotinas em Python para automação de download, controle de qualidade, análise e reconstrução espectral direcional de dados de ondas, utilizando como base histórica as estações oceanográficas do **National Data Buoy Center (NDBC/NOAA)**.

---

## Estrutura do Projeto e Scripts

O projeto foi modularizado em scripts independentes e especializados para garantir organização, reaproveitamento de dados e facilidade de manutenção:

1. **`download_arquivos.py`**
   - **Função**: Acessa automaticamente via internet o servidor histórico do NDBC/NOAA. Realiza o download dos arquivos compactados (`.txt.gz`) dos 5 parâmetros essenciais: densidade de energia (`w`), direções de propagação (`alpha1`, `alpha2`) e coeficientes de espalhamento de Fourier (`r1`, `r2`).
   - **Controle de Qualidade (QC)**: Filtra e limpa dados espúrios, substituindo os códigos de erro instrumentais (séries de 9) por valores nulos (`NaN`). Também corrige a escala física dos coeficientes $r_1$ e $r_2$, multiplicando-os por $0.01$. Ao final, consolida tudo em um único dicionário binário compactado (`.pkl`).

2. **`analise.py`**
   - **Função**: Carrega os dados limpos do arquivo `.pkl` e realiza o cálculo dos parâmetros integrados hora a hora para o ano inteiro.
   - **Cálculo de $H_s$**: Obtém a Altura Significativa calculando o momento de ordem zero ($m_0$), que representa a área total sob a curva do espectro de energia ($H_s = 4\sqrt{m_0}$).
   - **Cálculo de $T_p$**: Localiza a frequência de pico onde ocorre a maior densidade energética e extrai o Período de Pico através do seu inverso ($T_p = 1/f_p$).
   - **Saída**: Cria um índice cronológico unificado (`data_registro`) e exporta as séries temporais consolidadas em formato `.csv`.

3. **`histogramas.py`**
   - **Função**: Lê a série temporal em `.csv` e gera as distribuições de frequência estatística das variáveis marinhas.
   - **Análise Física**: Produz gráficos de barras independentes para identificar o clima de ondas padrão da estação, assimetrias temporais, limites de normalidade (faixas de 2.5m a 3.0m) e dominâncias dinâmicas de sistemas de vagas (*wind sea* de períodos curtos) e marulhos (*swell* de longos períodos). Salvando os outputs automaticamente em imagens `.png`.

4. **`espectro-teste.py`**
   - **Função**: Laboratório numérico de validação controlada. Cria um espectro direcional artificial utilizando uma distribuição Gaussiana na frequência e um espalhamento cossenoidal focado em uma direção (90° Leste).
   - **Objetivo**: Aplica a engenharia reversa para extrair os coeficientes e testa o comportamento intrínseco dos algoritmos frente a dados perfeitos, permitindo diagnosticar limitações teóricas como o fenômeno de *peak-splitting* e vazamento de energia.

5. **`espectro_direcional.py`**
   - **Função**: Script definitivo de reconstrução bidimensional $S(f, \theta)$ utilizando dados reais medidos pelas boias do NDBC.
   - **Modelagem**: Processa as matrizes de energia para cada uma das 47 frequências em todo o círculo trigonométrico (360 pontos) aplicando simultaneamente duas abordagens clássicas: a **Expansão Linear de Fourier** (2ª ordem truncada) e o método não-linear de alta resolução da **Máxima Entropia (MEM)**.
   - **Output**: Renderiza mapas de calor coloridos (`contourf` polar em coordenadas geográficas/náuticas) comparando a resolução de cada técnica.

---

## Fundamentação Teórica dos Métodos Reconstruídos

### Expansão da Série de Fourier (Truncada)
Assume que o espalhamento direcional pode ser aproximado por uma combinação linear suave de harmônicos senoidais. Devido às limitações físicas de medição das boias triaxiais, a série é cortada (**truncada**) na segunda ordem:
$$D_{\text{fourier}}(\theta) = \frac{1}{\pi} \left[ \frac{1}{2} + r_1\cos(\theta - \alpha_1) + r_2\cos(2(\theta - \alpha_2)) \right]$$
* **Limitações**: Apresenta baixa resolução angular (gerando borrões largos ao redor do pico) e sofre com oscilações numéricas que produzem **energias negativas artificiais**, violando leis físicas básicas.

### Método da Máxima Entropia (MEM)
Abordagem estatística baseada na entropia de Burg. Busca a distribuição direcional mais aleatória e menos tendenciosa possível para os componentes desconhecidos, vinculada estritamente aos 4 momentos medidos pela boia. É calculada de forma complexa resolvendo as equações de Yule-Walker:
$$D_{\text{MEM}}(\theta) = \frac{1}{2\pi} \frac{1 - \phi_1 c_1^* - \phi_2 c_2^*}{|1 - \phi_1 e^{-i\theta} - \phi_2 e^{-i2\theta}|^2}$$
* **Vantagens**: Proporciona altíssima resolução angular (picos extremamente estreitos e focados) e, devido ao módulo absoluto elevado ao quadrado no denominador, impede matematicamente o aparecimento de energias negativas, mantendo os resultados estritamente positivos ou nulos.

---

## Como Executar as Rotinas

Os códigos estão preparados para operar sequencialmente de forma automatizada. Garanta que o ambiente possua as bibliotecas `pandas`, `numpy` e `matplotlib` devidamente instaladas:

1. Altere o ID da boia e o ano desejado no cabeçalho do script de automação e execute o download/limpeza:
   ```bash
   python download_arquivos.py
   ```
2. Processe a matemática das séries temporais integradas de altura e período:
   ```bash
   python analise.py
   ```
3. Gere as análises gráficas estatísticas de histogramas:
   ```bash
   python histogramas.py
   ```
4. Para simular e validar os códigos no ambiente controlado do laboratório sintético:
   ```bash
   python espectro-teste.py
   ```
5. Para renderizar os mapas de calor polares e comparar os métodos direcionais com dados reais do oceano:
   ```bash
   python espectro_direcional.py
   ```
