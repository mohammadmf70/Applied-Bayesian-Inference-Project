import numpy as np
import matplotlib.pyplot as plt

n_flips = 250
actual_heads = 140
simulations = 1000

# Task 1: Fair Coin Model
fair_coin_results = np.random.binomial(n=n_flips, p=0.5, size=simulations)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(fair_coin_results, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(actual_heads, color='red', linestyle='dashed', linewidth=2, label=f'Actual: {actual_heads}')
plt.title("Fair Coin Model (p=0.5)")
plt.xlabel("Number of Heads")
plt.ylabel("Frequency")
plt.legend()

# Task 2: Bayesian Learner Model
p_samples = np.random.beta(141, 111, size=simulations)
bayesian_results = np.random.binomial(n=n_flips, p=p_samples)

plt.subplot(1, 2, 2)
plt.hist(bayesian_results, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.axvline(actual_heads, color='red', linestyle='dashed', linewidth=2, label=f'Actual: {actual_heads}')
plt.title("Bayesian Learner Model")
plt.xlabel("Number of Heads")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()
plt.show()

p_value = np.mean(fair_coin_results >= actual_heads)
print(f"Probability of seeing 140 heads or more in a fair coin world: {p_value:.4f}")