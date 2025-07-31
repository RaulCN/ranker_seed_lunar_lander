#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import csv

ENV_NAME = "LunarLander-v3"
N_SEEDS = 100
EPIS_PER_SEED = 10
CHECKPOINT_FREQ = 50

env = gym.make(ENV_NAME)
difficulty = {}

print("🚀 Avaliando NÍVEL DE DIFICULDADE das seeds...")
print("(quanto mais negativo = mais difícil)\n")

for seed in range(N_SEEDS):
    returns = []
    for _ in range(EPIS_PER_SEED):
        obs, _ = env.reset(seed=seed)
        done = truncated = False
        episode_r = 0
        while not (done or truncated):
            action = env.action_space.sample()
            obs, r, done, truncated, _ = env.step(action)
            episode_r += r
        returns.append(episode_r)
    difficulty[seed] = np.mean(returns)

    if (seed + 1) % 10 == 0:
        current_mean = np.mean(list(difficulty.values()))
        print(f"[{seed+1:3}/100] Sementes avaliadas. Média: {current_mean:.2f}")

env.close()

# Salvar CSV
with open("lunar_seeds_ranking_1.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["seed", "mean_reward"])
    for s, score in sorted(difficulty.items(), key=lambda x: x[1]):
        writer.writerow([s, round(score, 2)])

# =========================================================
# 1) RANKING DETALHADO
sorted_seeds = sorted(difficulty.items(), key=lambda x: x[1])
print("\n" + "=" * 70)
print("📚 CURRÍCULO DE DIFICULDADE – SEEDS ORDENADAS")
print("=" * 70)
print("Pos  Seed   Dificuldade    Fase do Treino")
print("-" * 70)

for rank, (s, score) in enumerate(sorted_seeds, start=1):
    if rank <= 25:
        phase = "🟢 INICIAL"
    elif rank <= 50:
        phase = "🟡 INTERM."
    elif rank <= 75:
        phase = "🟠 AVANÇ."
    else:
        phase = "🔴 FINAL"
    print(f"{rank:>3}º  {s:>4}   {round(score, 2):>9}   {phase}")

# =========================================================
# 2) TABELA RESUMO FÁCIL / MÉDIO / DIFÍCIL
print("\n" + "=" * 60)
print("📊 RESUMO – DIVISÃO FÁCIL · MÉDIO · DIFÍCIL")
print("=" * 60)

easy   = sorted_seeds[:34]   # ~1/3 mais fáceis
medium = sorted_seeds[34:67] # ~1/3 médias
hard   = sorted_seeds[67:]   # ~1/3 mais difíceis

print(f"\n🟢 FÁCIL   ({len(easy)} seeds)   – use no início do currículo")
print("   Seeds:", ", ".join(str(s) for s, _ in easy))

print(f"\n🟡 MÉDIO   ({len(medium)} seeds)  – use na metade do currículo")
print("   Seeds:", ", ".join(str(s) for s, _ in medium))

print(f"\n🔴 DIFÍCIL ({len(hard)} seeds)  – use só no final do currículo")
print("   Seeds:", ", ".join(str(s) for s, _ in hard))

print("\n" + "=" * 60)
print("💡 Copie/cole as listas acima no seu pipeline de treino!")
print("CSV salvo: lunar_seeds_ranking_1.csv ✅")
