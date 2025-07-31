# ranker_seed_lunar_lander

Este projeto implementa um método de **Curriculum Learning** adaptativo para treinar um agente de Deep Reinforcement Learning (DRL) no ambiente `LunarLander-v3` do **Gymnasium**. O agente é treinado usando o algoritmo **PPO** (*Proximal Policy Optimization*) da biblioteca **Stable Baselines3**.

A abordagem principal é **ranquear as sementes (seeds)** do ambiente para identificar os cenários de pouso mais fáceis e mais difíceis. O agente começa a treinar nos cenários mais fáceis e, à medida que sua performance melhora, ele é gradualmente exposto a ambientes mais complexos, acelerando e estabilizando o aprendizado.

---

## 📌 Funcionalidades

- **Ranking de Sementes**  
  Script separado (`ranking_seeds.py`) avalia 100 sementes do ambiente `LunarLander-v3` para criar um ranking de dificuldade.  
  Isso gera o arquivo `lunar_seeds_ranking.csv`, base para o curriculum.

- **Curriculum Learning Adaptativo**  
  O treinamento é dividido em fases. O agente começa em um ambiente simplificado e progride para cenários com mais vento e turbulência.

- **Otimização de Hiperparâmetros Seguros**  
  Agendadores de aprendizado (`learning_rate` e `clip_range`) ajustam-se dinamicamente ao longo do treinamento, mantendo o processo mais estável.

- **Mecanismo de Regressão Controlada**  
  Se a performance cair drasticamente em uma fase mais difícil, o ambiente é temporariamente facilitado para permitir recuperação, evitando "desaprendizado".

- **Customização do Ambiente com Wrappers**  
  Utiliza *wrappers* como `LunarLanderAdaptiveWrapper`, `HarderLanderWrapper`, `EasierLanderWrapper` e `StepPenaltyWrapper` para modificar dinamicamente dificuldade e recompensa em tempo real.

---

## ⚙️ Pré-requisitos

- Python 3.8+
- `gymnasium`
- `stable_baselines3`
- `pandas`
- `numpy`
- `tqdm` *(opcional, para barra de progresso no script de ranking)*

Instale as dependências com:

```bash
pip install gymnasium stable-baselines3 pandas numpy tqdm

## Executando ranking_seeds_dificuldade.py

saída no console~

= RESTART: /home/rauto/Área de trabalho/L lander/ranking seeds l lander/novo/ranking_seeds_1 (cópia).py
🚀 Avaliando NÍVEL DE DIFICULDADE das seeds...
(quanto mais negativo = mais difícil)

[ 10/100] Sementes avaliadas. Média: -170.19
[ 20/100] Sementes avaliadas. Média: -184.58
[ 30/100] Sementes avaliadas. Média: -185.06
[ 40/100] Sementes avaliadas. Média: -178.44
[ 50/100] Sementes avaliadas. Média: -176.18
[ 60/100] Sementes avaliadas. Média: -179.05
[ 70/100] Sementes avaliadas. Média: -182.95
[ 80/100] Sementes avaliadas. Média: -179.76
[ 90/100] Sementes avaliadas. Média: -178.64
[100/100] Sementes avaliadas. Média: -178.30

======================================================================
📚 CURRÍCULO DE DIFICULDADE – SEEDS ORDENADAS
======================================================================
Pos  Seed   Dificuldade    Fase do Treino
----------------------------------------------------------------------
  1º    85      -318.0   🟢 INICIAL
  2º     6     -296.03   🟢 INICIAL
  3º    29     -287.07   🟢 INICIAL
  4º    68     -283.98   🟢 INICIAL
  5º    57     -280.59   🟢 INICIAL
  6º    65     -275.79   🟢 INICIAL
  7º    67     -273.78   🟢 INICIAL
  8º    16     -273.32   🟢 INICIAL
  9º    17     -270.32   🟢 INICIAL
 10º    81     -267.25   🟢 INICIAL
 11º    26     -257.49   🟢 INICIAL
 12º    69     -255.09   🟢 INICIAL
 13º    27     -253.45   🟢 INICIAL
 14º    32      -252.8   🟢 INICIAL
 15º    18     -249.33   🟢 INICIAL
 16º    49     -248.61   🟢 INICIAL
 17º    44     -246.39   🟢 INICIAL
 18º    21     -238.09   🟢 INICIAL
 19º    13     -237.52   🟢 INICIAL
 20º    95     -231.52   🟢 INICIAL
 21º    63      -230.2   🟢 INICIAL
 22º    66      -226.2   🟢 INICIAL
 23º    93      -218.1   🟢 INICIAL
 24º    51     -216.97   🟢 INICIAL
 25º    77     -212.32   🟢 INICIAL
 26º     1     -211.84   🟡 INTERM.
 27º    30     -210.97   🟡 INTERM.
 28º    97     -208.99   🟡 INTERM.
 29º    23      -208.6   🟡 INTERM.
 30º    74     -205.57   🟡 INTERM.
 31º    55     -205.04   🟡 INTERM.
 32º     9     -204.43   🟡 INTERM.
 33º    75     -201.59   🟡 INTERM.
 34º    54     -200.39   🟡 INTERM.
 35º    41     -199.37   🟡 INTERM.
 36º    59     -198.67   🟡 INTERM.
 37º    38     -198.21   🟡 INTERM.
 38º    58     -197.82   🟡 INTERM.
 39º    99     -195.69   🟡 INTERM.
 40º    47     -195.25   🟡 INTERM.
 41º    10     -187.04   🟡 INTERM.
 42º    15     -185.91   🟡 INTERM.
 43º    56     -184.87   🟡 INTERM.
 44º    91     -183.53   🟡 INTERM.
 45º    53     -178.09   🟡 INTERM.
 46º    60     -177.24   🟡 INTERM.
 47º    78      -177.2   🟡 INTERM.
 48º    36     -175.94   🟡 INTERM.
 49º    96     -172.03   🟡 INTERM.
 50º    42     -171.67   🟡 INTERM.
 51º    90     -171.19   🟠 AVANÇ.
 52º     3     -169.75   🟠 AVANÇ.
 53º    19     -169.62   🟠 AVANÇ.
 54º    25     -166.81   🟠 AVANÇ.
 55º    89     -164.79   🟠 AVANÇ.
 56º    12     -163.98   🟠 AVANÇ.
 57º    86     -160.07   🟠 AVANÇ.
 58º    35      -158.8   🟠 AVANÇ.
 59º    33     -157.77   🟠 AVANÇ.
 60º     2     -157.69   🟠 AVANÇ.
 61º     5     -157.49   🟠 AVANÇ.
 62º    87     -153.03   🟠 AVANÇ.
 63º    52     -150.86   🟠 AVANÇ.
 64º    14     -149.89   🟠 AVANÇ.
 65º    45     -148.39   🟠 AVANÇ.
 66º    73     -146.39   🟠 AVANÇ.
 67º    43      -141.2   🟠 AVANÇ.
 68º    61     -141.06   🟠 AVANÇ.
 69º    22     -140.49   🟠 AVANÇ.
 70º    71     -139.59   🟠 AVANÇ.
 71º    39     -139.24   🟠 AVANÇ.
 72º    83     -138.39   🟠 AVANÇ.
 73º    94     -138.23   🟠 AVANÇ.
 74º    79     -137.13   🟠 AVANÇ.
 75º    72     -133.99   🟠 AVANÇ.
 76º    48      -133.6   🔴 FINAL
 77º     0     -133.36   🔴 FINAL
 78º    82     -132.69   🔴 FINAL
 79º    84     -131.77   🔴 FINAL
 80º    98     -130.16   🔴 FINAL
 81º     7     -127.17   🔴 FINAL
 82º     8      -126.5   🔴 FINAL
 83º    80     -125.28   🔴 FINAL
 84º    76     -124.38   🔴 FINAL
 85º    50     -120.68   🔴 FINAL
 86º    28     -120.08   🔴 FINAL
 87º    37     -118.51   🔴 FINAL
 88º     4     -117.61   🔴 FINAL
 89º    34     -109.58   🔴 FINAL
 90º    20     -107.97   🔴 FINAL
 91º    88     -105.54   🔴 FINAL
 92º    92     -103.21   🔴 FINAL
 93º    11     -102.82   🔴 FINAL
 94º    62     -102.67   🔴 FINAL
 95º    46      -99.75   🔴 FINAL
 96º    64      -97.21   🔴 FINAL
 97º    70      -95.97   🔴 FINAL
 98º    40      -87.45   🔴 FINAL
 99º    24      -80.23   🔴 FINAL
100º    31       -63.7   🔴 FINAL

============================================================
📊 RESUMO – DIVISÃO FÁCIL · MÉDIO · DIFÍCIL
============================================================

🟢 FÁCIL   (34 seeds)   – use no início do currículo
   Seeds: 85, 6, 29, 68, 57, 65, 67, 16, 17, 81, 26, 69, 27, 32, 18, 49, 44, 21, 13, 95, 63, 66, 93, 51, 77, 1, 30, 97, 23, 74, 55, 9, 75, 54

🟡 MÉDIO   (33 seeds)  – use na metade do currículo
   Seeds: 41, 59, 38, 58, 99, 47, 10, 15, 56, 91, 53, 60, 78, 36, 96, 42, 90, 3, 19, 25, 89, 12, 86, 35, 33, 2, 5, 87, 52, 14, 45, 73, 43

🔴 DIFÍCIL (33 seeds)  – use só no final do currículo
   Seeds: 61, 22, 71, 39, 83, 94, 79, 72, 48, 0, 82, 84, 98, 7, 8, 80, 76, 50, 28, 37, 4, 34, 20, 88, 92, 11, 62, 46, 64, 70, 40, 24, 31

============================================================
💡 Copie/cole as listas acima no seu pipeline de treino!
CSV salvo: lunar_seeds_ranking_1.csv ✅

