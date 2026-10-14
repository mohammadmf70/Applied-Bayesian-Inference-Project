import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.ndimage import gaussian_filter

np.random.seed(0)
x = np.linspace(-2, 3, 10)
m_true, c_true, sigma = 1.5, 2.0, 0.5
y = m_true * x + c_true + np.random.normal(0, sigma, len(x))

m_vals = np.linspace(0, 3, 50)
c_vals = np.linspace(-2, 5, 50)
M, C = np.meshgrid(m_vals, c_vals)

posterior = np.ones_like(M)
posterior = posterior / posterior.sum()

for xi, yi in zip(x, y):
    likelihood = norm.pdf(yi, loc=M * xi + C, scale=sigma)
    posterior *= likelihood
    posterior /= posterior.sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

im = ax1.imshow(posterior, extent=[0, 3, -2, 5], origin='lower',
                aspect='auto', cmap='hot', interpolation='bilinear')
ax1.set_xlabel('Slope (m)')
ax1.set_ylabel('Intercept (c)')
ax1.set_title('Left Panel: Final Posterior Distribution')
fig.colorbar(im, ax=ax1, label='Probability')

max_idx = np.unravel_index(posterior.argmax(), posterior.shape)
m_map, c_map = M[max_idx], C[max_idx]
ax1.plot(m_true, c_true, 'bo', markersize=12, label='True parameters')
ax1.plot(m_map, c_map, 'r*', markersize=15, label='MAP estimate')
ax1.legend()

smoothed = gaussian_filter(posterior, sigma=1.0)
threshold = np.percentile(smoothed, 68)
ax1.contour(M, C, smoothed, levels=[threshold], colors='white', linewidths=2, linestyles='--')

n_samples = 50
posterior_flat = posterior.flatten()
posterior_flat = posterior_flat / posterior_flat.sum()
indices = np.random.choice(len(posterior_flat), size=n_samples, p=posterior_flat)

m_samples, c_samples = [], []
for idx in indices:
    i, j = np.unravel_index(idx, M.shape)
    m_samples.append(M[i, j])
    c_samples.append(C[i, j])

ax2.scatter(x, y, color='red', s=120, label='Observed data', edgecolors='black', zorder=5)
x_line = np.linspace(-2, 3, 100)
for m, c in zip(m_samples, c_samples):
    ax2.plot(x_line, m * x_line + c, 'b', alpha=0.1, linewidth=1)

ax2.plot(x_line, m_true * x_line + c_true, 'g', linewidth=3, label='True line')
y_samples = np.array([m * x_line + c for m, c in zip(m_samples, c_samples)])
y_upper = np.percentile(y_samples, 97.5, axis=0)
y_lower = np.percentile(y_samples, 2.5, axis=0)
ax2.fill_between(x_line, y_lower, y_upper, alpha=0.2, color='blue', label='95% CI')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Right Panel: Sampled Lines from Posterior')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.suptitle('Bayesian Linear Regression Results (Grid Approximation)')
plt.tight_layout()
plt.show()