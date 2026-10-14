import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd

movies_posterior = {
    'A (Unknown)': {'alpha': 4, 'beta': 2},
    'B (Popular)': {'alpha': 92, 'beta': 12},
    'C (Controversial)': {'alpha': 52, 'beta': 22}
}

results = []
for movie, params in movies_posterior.items():
    alpha = params['alpha']
    beta_val = params['beta']
    mean = alpha / (alpha + beta_val)
    percentile_5 = beta.ppf(0.05, alpha, beta_val)
    results.append({
        'Movie': movie,
        'Alpha': alpha,
        'Beta': beta_val,
        'Mean': mean,
        '5th_Percentile': percentile_5
    })

df = pd.DataFrame(results)
print("Ranking by Mean:")
print(df.sort_values('Mean', ascending=False)[['Movie', 'Mean']])

print("\nRanking by 5th Percentile (Lower Bound):")
print(df.sort_values('5th_Percentile', ascending=False)[['Movie', '5th_Percentile']])

plt.figure(figsize=(10, 5))
x = np.linspace(0, 1, 1000)
colors = {'A (Unknown)': 'red', 'B (Popular)': 'green', 'C (Controversial)': 'blue'}

for movie, params in movies_posterior.items():
    a, b = params['alpha'], params['beta']
    y_vals = beta.pdf(x, a, b)
    plt.plot(x, y_vals, color=colors[movie], linewidth=2, label=movie)
    p5 = beta.ppf(0.05, a, b)
    plt.axvline(p5, color=colors[movie], linestyle='--', alpha=0.7)
    plt.fill_between(x[x <= p5], y_vals[x <= p5], alpha=0.3, color=colors[movie])

plt.xlabel('Rating')
plt.ylabel('Density')
plt.title('Posterior Distributions with 5th Percentile (Lower Bound)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()