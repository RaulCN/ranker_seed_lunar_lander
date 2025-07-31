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
