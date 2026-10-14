import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

clicks_A, visits_A = 100, 1200
clicks_B, visits_B = 110, 1200

samples_A = beta.rvs(clicks_A + 1, visits_A - clicks_A + 1, size=100000)
samples_B = beta.rvs(clicks_B + 1, visits_B - clicks_B + 1, size=100000)

diff = samples_B - samples_A
prob_B_better = (diff > 0).mean()

loss_values = diff[diff < 0]
expected_loss = np.abs(loss_values.mean()) if len(loss_values) > 0 else 0

print("--- Bayesian Analysis Results ---")
print(f"Probability of Campaign B being superior: {prob_B_better * 100:.2f}%")
print(f"Expected loss (if B is chosen incorrectly): {expected_loss:.5f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(samples_A, bins=50, alpha=0.5, label='Campaign A', color='blue')
plt.hist(samples_B, bins=50, alpha=0.5, label='Campaign B', color='orange')
plt.title('Posterior Distributions of A and B')
plt.xlabel('Conversion Rate')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(diff, bins=50, alpha=0.7, color='green')
plt.axvline(0, color='red', linestyle='--')
plt.title('Distribution of Difference (B - A)')
plt.xlabel('Delta')

plt.tight_layout()
plt.show()