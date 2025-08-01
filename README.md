# 🧠 ranker_seed_lunar_lander

Este projeto tem como **objetivo central** o **rankeamento de seeds** do ambiente `LunarLander-v3` (da biblioteca **Gymnasium**) com base em sua **dificuldade objetiva**, medida pelo desempenho médio de um agente PPO pré-treinado.

🔍 Em vez de utilizar seeds aleatórias nos treinamentos ou testes, este projeto fornece um **critério estruturado** para selecionar ambientes com diferentes níveis de dificuldade (fácil, médio e difícil), permitindo:

- Curadoria de currículos de treino (opcional)
- Benchmarking consistente entre agentes
- Testes controlados de generalização
- Geração de ambientes desafiadores sob demanda

---

## 🚀 O que o script faz?

O script `ranking_seeds_dificuldade.py`:

1. Avalia automaticamente **100 seeds** do ambiente `LunarLander-v3`.
2. Mede a **recompensa média** para cada seed ao rodar um agente PPO pré-treinado.
3. Ordena as seeds da mais difícil (menor recompensa média) para a mais fácil.
4. Divide as seeds em 3 grupos:
   - 🟢 FÁCIL
   - 🟡 MÉDIO
   - 🔴 DIFÍCIL
5. Salva os resultados em um CSV: `lunar_seeds_ranking.csv`

---

## 📁 Estrutura do Projeto

```
├── ranking_seeds_dificuldade.py   # Script principal de avaliação e ranqueamento
├── lunar_seeds_ranking.csv        # Arquivo gerado com o ranking das seeds
└── ppo_model.zip                  # Agente PPO utilizado para avaliação
```

---

## 🧪 Exemplo de Saída no Console

```text
🚀 Avaliando NÍVEL DE DIFICULDADE das seeds...
(quanto mais negativo = mais difícil)

[100/100] Sementes avaliadas. Média: -178.30

======================================================================
📚 CURRÍCULO DE DIFICULDADE – SEEDS ORDENADAS
======================================================================
Pos  Seed   Dificuldade    Fase do Treino
----------------------------------------------------------------------
  1º    85      -318.0   🟢 INICIAL
  2º     6     -296.03   🟢 INICIAL
  ...
 99º    24      -80.23   🔴 FINAL
100º    31       -63.7   🔴 FINAL

============================================================
📊 RESUMO – DIVISÃO FÁCIL · MÉDIO · DIFÍCIL
============================================================

🟢 FÁCIL   (34 seeds)
   Seeds: 85, 6, 29, 68, 57, 65, 67, ...

🟡 MÉDIO   (33 seeds)
   Seeds: 41, 59, 38, 58, 99, 47, ...

🔴 DIFÍCIL (33 seeds)
   Seeds: 61, 22, 71, 39, 83, 94, ...
```

---

## ⚙️ Requisitos

- Python 3.8+
- `gymnasium`
- `stable-baselines3`
- `numpy`
- `pandas`
- `tqdm` (opcional)

Instale via:

```bash
pip install gymnasium stable-baselines3 pandas numpy tqdm
```

---

## 🧠 Aplicações

- **Treinamento com currículo programado**
- **Criação de testes robustos e balanceados**
- **Estudos de generalização e transferência**
- **Pesquisa em dificuldade adaptativa**

---

## 📎 Referência

Ambiente: `LunarLander-v3`  
Modelo PPO base: `Stable-Baselines3`  
Autor: Raul Campos Nascimento  
