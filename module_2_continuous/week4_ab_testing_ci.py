import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

clicks_A, views_A = 100, 1200
clicks_B, views_B = 110, 1200

alpha_A, beta_A = 1 + clicks_A, 1 + (views_A - clicks_A)
alpha_B, beta_B = 1 + clicks_B, 1 + (views_B - clicks_B)

n_samples = 100000
np.random.seed(42)

samples_A = np.random.beta(alpha_A, beta_A, n_samples)
samples_B = np.random.beta(alpha_B, beta_B, n_samples)

prob_B_better = np.mean(samples_B > samples_A)
print(f"P(B > A) = {prob_B_better * 100:.2f}%")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
x = np.linspace(0.06, 0.11, 100)
plt.plot(x, beta.pdf(x, alpha_A, beta_A), 'r', linewidth=2, label='Campaign A')
plt.plot(x, beta.pdf(x, alpha_B, beta_B), 'b', linewidth=2, label='Campaign B')
plt.xlabel('CTR')
plt.ylabel('Density')
plt.title('Posterior Distributions')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
diff = samples_B - samples_A
plt.hist(diff * 100, bins=50, density=True, alpha=0.7, color='purple', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No difference')
plt.axvline(x=np.percentile(diff, 5) * 100, color='green', linestyle=':', linewidth=2, label='5th percentile')
plt.xlabel('Difference (B - A) in percentage points')
plt.ylabel('Density')
plt.title(f'Distribution of Difference\nP(B > A) = {prob_B_better * 100:.1f}%')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()