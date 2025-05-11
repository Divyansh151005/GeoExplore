import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Create a directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# Generate mineral potential maps
print("Generating mineral potential maps...")

# Set up common grid
x = np.linspace(74.0, 80.0, 100)
y = np.linspace(11.0, 19.0, 100)
X, Y = np.meshgrid(x, y)

# 1. Generate Gold potential map
plt.figure(figsize=(12, 8))
# Create synthetic mineral potential data
Z2 = np.zeros_like(X)

# Create potential hotspots for Gold
gold_centers = [(77.5, 15.8), (76.3, 14.5), (75.7, 12.9)]
for cx, cy in gold_centers:
    Z2 += 0.8 * np.exp(-0.3*((X-cx)**2 + (Y-cy)**2))
    
# Add some randomness
Z2 += np.random.normal(0, 0.05, Z2.shape)
Z2 = np.clip(Z2, 0, 1)
Z2 = gaussian_filter(Z2, sigma=2)

# Plot contour
contour = plt.contourf(X, Y, Z2, cmap='YlOrRd', levels=15)
plt.colorbar(label='Gold Potential (Probability)')
# Add contour lines for high probability areas
high_prob = plt.contour(X, Y, Z2, levels=[0.7], colors='red', linewidths=2)
plt.title('Gold Mineral Potential Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.savefig("visualizations/gold_potential_map.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Generate REE potential map
plt.figure(figsize=(12, 8))
# Create synthetic mineral potential data for REE
Z3 = np.zeros_like(X)

# Create potential hotspots for REE
ree_centers = [(78.8, 16.2), (77.9, 15.1), (76.5, 13.6)]
for cx, cy in ree_centers:
    Z3 += 0.7 * np.exp(-0.25*((X-cx)**2 + (Y-cy)**2))
    
# Add some randomness
Z3 += np.random.normal(0, 0.05, Z3.shape)
Z3 = np.clip(Z3, 0, 1)
Z3 = gaussian_filter(Z3, sigma=2)

# Plot contour
contour = plt.contourf(X, Y, Z3, cmap='Greens', levels=15)
plt.colorbar(label='REE Potential (Probability)')
# Add contour lines for high probability areas
high_prob = plt.contour(X, Y, Z3, levels=[0.7], colors='darkgreen', linewidths=2)
plt.title('Rare Earth Elements (REE) Potential Map')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.savefig("visualizations/ree_potential_map.png", dpi=300, bbox_inches='tight')
plt.close()

print("Mineral potential maps generated successfully.")