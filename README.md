# Processamento Espectral e Análise Estatística de Dados de Ondas

Repositório dedicado ao desenvolvimento de rotinas em Python para automação de download, controle de qualidade, análise e reconstrução espectral direcional de dados de ondas, utilizando como base histórica as estações oceanográficas do **National Data Buoy Center (NDBC/NOAA)**.

---

## Estrutura do Projeto e Scripts

O projeto foi modularizado em scripts independentes e especializados para garantir organização, reaproveitamento de dados e facilidade de manutenção:

1. **`download_arquivos.py`**
   - **Função**: Acessa automaticamente via internet o servidor histórico do NDBC/NOAA. Realiza o download dos arquivos compactados (`.txt.gz`) dos 5 parâmetros essenciais: densidade de energia (`w`), direções de propagação (`alpha1`, `alpha2`) e coeficientes de espalhamento de Fourier (`r1`, `r2`). Arquivos já existentes na pasta local não são baixados novamente.
   - **Controle de Qualidade (QC)**: Corrige a escala física dos coeficientes $r_1$ e $r_2$, multiplicando-os por $0.01$ (os arquivos originais armazenam esses valores em centésimos). Ao final, consolida tudo em um único dicionário binário compactado (`.pkl`).
   - **Uso como módulo**: Toda a rotina está encapsulada na função `baixar_dados(id_boia, year)`, que retorna o caminho do `.pkl` gerado ou `None` caso algum dos 5 arquivos não exista no servidor. Isso permite que outros scripts baixem automaticamente anos que ainda não estejam disponíveis localmente.

2. **`wind.py`**
   - **Função**: Realiza o download dos dados meteorológicos padrão da boia (pasta `stdmet` do NDBC, arquivos com sufixo `h`), contendo a direção (`WDIR`) e a velocidade (`WSPD`) do vento local.
   - **Controle de Qualidade (QC)**: Substitui os códigos de ausência de medição (`999` para direção e `99.0` para velocidade) por valores nulos (`NaN`).
   - **Saída**: Cria o índice cronológico `data_registro` e exporta as séries de vento em formato `.csv` (`wind_{boia}_{ano}.csv`), utilizadas pelo `espectro_direcional.py`.

3. **`analise.py`**
   - **Função**: Carrega os dados limpos do arquivo `.pkl` e realiza o cálculo dos parâmetros integrados registro a registro para o ano inteiro.
   - **Cálculo de $H_s$**: Obtém a Altura Significativa calculando o momento de ordem zero ($m_0$), que representa a área total sob a curva do espectro de energia ($H_s = 4\sqrt{m_0}$), utilizando a largura de cada faixa de frequência (`np.diff`).
   - **Cálculo de $T_p$**: Localiza a frequência de pico onde ocorre a maior densidade energética e extrai o Período de Pico através do seu inverso ($T_p = 1/f_p$).
   - **Auditoria Temporal**: Remove registros duplicados e reindexa a série sobre um calendário completo no intervalo de amostragem da boia, expondo os horários sem registro como `NaN` para a contagem de falhas (*gaps*).
   - **Saída**: Cria um índice cronológico unificado (`data_registro`) e exporta as séries temporais consolidadas em formato `.csv`.

4. **`histogramas.py`**
   - **Função**: Lê a série temporal em `.csv` e gera as distribuições de frequência estatística das variáveis marinhas.
   - **Análise Física**: Produz gráficos de barras independentes para identificar o clima de ondas padrão da estação, assimetrias temporais, limites de normalidade (faixas de 2.5m a 3.0m) e dominâncias dinâmicas de sistemas de vagas (*wind sea* de períodos curtos) e marulhos (*swell* de longos períodos). Salvando os outputs automaticamente em imagens `.png`.
   - **Estatísticas de $H_s$**: Marca no histograma a mediana, o valor máximo e o percentil 99 ($P_{99}$, valor de $H_s$ ultrapassado em apenas 1% dos registros), que descreve o mar de tempestade do local de forma mais robusta que o máximo, por não depender de um único evento.
   - **Escala do Eixo Vertical**: Utiliza escala de raiz quadrada no eixo das contagens, comprimindo as classes com milhares de registros mais do que as classes com poucas dezenas. Isso torna visível a cauda da distribuição (eventos extremos), mantendo no eixo os valores reais de contagem.
   - **Verificação de Extremos**: A função `checar_vizinhos` localiza o registro de $H_s$ máximo e exibe os registros vizinhos, permitindo verificar se o valor pertence a um evento contínuo (tempestade) ou se é um valor isolado e possivelmente espúrio.

5. **`espectro-teste.py`**
   - **Função**: Laboratório numérico de validação controlada. Cria um espectro direcional artificial utilizando uma distribuição Gaussiana na frequência e um espalhamento cossenoidal focado em uma direção (90° Leste).
   - **Objetivo**: Aplica a engenharia reversa para extrair os coeficientes e testa o comportamento intrínseco dos algoritmos frente a dados perfeitos, permitindo diagnosticar limitações teóricas como o fenômeno de *peak-splitting* e vazamento de energia.

6. **`espectro_direcional.py`**
   - **Função**: Script de reconstrução bidimensional $S(f, \theta)$ para um instante específico (data e hora definidas em `data_alvo`), utilizando dados reais medidos pelas boias do NDBC. Os espectros são registrados nos minutos 10 e 40 de cada hora.
   - **Modelagem**: Processa as matrizes de energia para cada uma das 47 frequências em todo o círculo trigonométrico (360 pontos) aplicando simultaneamente duas abordagens clássicas: a **Expansão Linear de Fourier** (2ª ordem truncada) e o método não-linear de alta resolução da **Máxima Entropia (MEM)**.
   - **Integração com o Vento**: Como os registros de vento e de espectro não ocorrem nos mesmos minutos, o vento correspondente é localizado por proximidade temporal com `pd.merge_asof` (direção `nearest`, tolerância de 20 minutos). Caso não exista vento válido dentro da tolerância, o script informa a ausência.
   - **Output**: Renderiza mapas de calor coloridos (`contourf` polar em coordenadas geográficas/náuticas) comparando a resolução de cada técnica, com a direção e a velocidade do vento indicadas na figura. Gera também um gráfico cartesiano de $D(\theta)$ na frequência de pico, sobrepondo os dois métodos e uma linha vertical na direção do vento.

7. **`espectro_direcional_data_especifica.py`**
   - **Função**: Reconstrução bidimensional $S(f, \theta)$ da média diária de um dia escolhido (`dia_alvo` e `mes_alvo`), aplicando os métodos de Fourier e MEM.
   - **Média dos Momentos**: A média é calculada sobre os momentos complexos de cada registro ($c_1 = r_1e^{i\alpha_1}$ e $c_2 = r_2e^{2i\alpha_2}$), ponderados pela densidade de energia `w`. Registros com valores nulos recebem peso zero. Os detalhes estão na seção de fundamentação teórica.
   - **Verificação**: Exibe a integral de $D(\theta)$ em cada frequência pela regra dos trapézios (`np.trapezoid`), que deve resultar em 1 para ambos os métodos.
   - **Output**: Gera os mapas polares da média diária e o gráfico cartesiano de $D(\theta)$ na frequência de pico.

8. **`espectro_direcional_estacoes.py`**
   - **Função**: Reconstrução bidimensional $S(f, \theta)$ média para cada estação do ano do Hemisfério Norte (Inverno, Primavera, Verão e Outono), utilizando a mesma média ponderada dos momentos complexos do script diário.
   - **Inverno**: O inverno de um ano $Y$ é composto por dezembro de $Y-1$, janeiro e fevereiro de $Y$. Dezembro de $Y$ pertence ao inverno seguinte e não é incluído. Caso os dados de $Y-1$ não estejam disponíveis localmente, o script tenta baixá-los com `baixar_dados`. Se o ano anterior não existir no servidor, o inverno é calculado apenas com janeiro e fevereiro, e o título da figura indica essa limitação.
   - **Output**: Gera um par de mapas polares (Fourier e MEM) para cada estação, com o período utilizado indicado no título.

---

## Fundamentação Teórica dos Métodos Reconstruídos

### Expansão da Série de Fourier (Truncada)
Assume que o espalhamento direcional pode ser aproximado por uma combinação linear suave de harmônicos senoidais. Devido às limitações físicas de medição das boias triaxiais, a série é cortada (**truncada**) na segunda ordem:
$$D_{\text{fourier}}(\theta) = \frac{1}{\pi} \left[ \frac{1}{2} + r_1\cos(\theta - \alpha_1) + r_2\cos(2(\theta - \alpha_2)) \right]$$
* **Limitações**: Apresenta baixa resolução angular (gerando borrões largos ao redor do pico) e sofre com oscilações numéricas que produzem **energias negativas artificiais**, violando leis físicas básicas. Por ser linear, sua integral é sempre igual a 1, mesmo quando os momentos de entrada são inconsistentes, de modo que o método não sinaliza esse tipo de problema.

### Método da Máxima Entropia (MEM)
Abordagem estatística baseada na entropia de Burg. Busca a distribuição direcional mais aleatória e menos tendenciosa possível para os componentes desconhecidos, vinculada estritamente aos 4 momentos medidos pela boia. É calculada de forma complexa resolvendo as equações de Yule-Walker:
$$D_{\text{MEM}}(\theta) = \frac{1}{2\pi} \frac{1 - \phi_1 c_1^* - \phi_2 c_2^*}{|1 - \phi_1 e^{-i\theta} - \phi_2 e^{-i2\theta}|^2}$$
* **Vantagens**: Proporciona altíssima resolução angular (picos extremamente estreitos e focados) e reproduz exatamente os momentos medidos. O denominador, por ser um módulo elevado ao quadrado, nunca é negativo.
* **Limitações**: O sinal de $D_{\text{MEM}}$ é determinado pelo numerador. Quando os momentos não são fisicamente realizáveis, isto é, quando nenhuma distribuição direcional real poderia produzi-los ($|c_2 - c_1^2| > 1 - |c_1|^2$), o numerador torna-se negativo e a curva inteira é invertida, com integral igual a $-1$. Além disso, para espalhamentos unimodais com $r_2$ baixo em relação a $r_1$, o método pode dividir um único pico em dois (*peak-splitting*), de modo que um pico duplo no MEM não comprova, isoladamente, a existência de um mar bimodal.

### Média Temporal dos Parâmetros Direcionais
Os parâmetros $r$ e $\alpha$ não são independentes: juntos formam os momentos complexos $c_1 = r_1e^{i\alpha_1}$ e $c_2 = r_2e^{2i\alpha_2}$. Calcular a média de $r$ e de $\alpha$ separadamente superestima a concentração direcional quando a direção varia ao longo do período, o que pode gerar momentos não realizáveis. Por isso, as médias diária e sazonal são calculadas sobre os momentos complexos, ponderados pela densidade de energia:
$$\bar{c}_n = \frac{\sum w \, c_n}{\sum w}$$
Esse procedimento equivale a calcular a média do espectro direcional $S(f, \theta)$ e garante que os momentos médios sejam sempre realizáveis.

---

## Como Executar as Rotinas

Os códigos estão preparados para operar sequencialmente de forma automatizada. Garanta que o ambiente possua as bibliotecas `pandas`, `numpy` e `matplotlib` devidamente instaladas. Em cada script, o ID da boia e o ano são definidos no cabeçalho (`id_boia` e `year`). Todos os dados são salvos e lidos na pasta `dados/{boia}_{ano}/`, relativa à pasta de onde os scripts são executados:

1. Altere o ID da boia e o ano desejado no cabeçalho do script de automação e execute o download/limpeza:
   ```bash
   python download_arquivos.py
   ```
2. Baixe e trate os dados de vento da mesma boia e ano:
   ```bash
   python wind.py
   ```
3. Processe a matemática das séries temporais integradas de altura e período:
   ```bash
   python analise.py
   ```
4. Gere as análises gráficas estatísticas de histogramas:
   ```bash
   python histogramas.py
   ```
5. Para simular e validar os códigos no ambiente controlado do laboratório sintético:
   ```bash
   python espectro-teste.py
   ```
6. Para renderizar os mapas de calor polares e comparar os métodos direcionais com dados reais do oceano em um instante específico:
   ```bash
   python espectro_direcional.py
   ```
7. Para a reconstrução direcional média de um dia específico:
   ```bash
   python espectro_direcional_data_especifica.py
   ```
8. Para a reconstrução direcional média de cada estação do ano:
   ```bash
   python espectro_direcional_estacoes.py
   ```
