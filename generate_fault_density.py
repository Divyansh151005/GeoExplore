import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter

# Create a directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# Generate fault density heatmap
print("Generating fault density heatmap...")
plt.figure(figsize=(10, 8))
# Create synthetic fault density data
x = np.linspace(74.0, 80.0, 100)
y = np.linspace(11.0, 19.0, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Create some "hotspots" of fault density
centers = [(77.2, 15.3), (76.1, 14.2), (78.5, 16.8), (75.4, 12.6)]
for cx, cy in centers:
    Z += np.exp(-0.2*((X-cx)**2 + (Y-cy)**2))
    
Z = gaussian_filter(Z, sigma=2)

# Plot contour
plt.contourf(X, Y, Z, cmap='YlOrRd', levels=15)
plt.colorbar(label='Fault Density')
plt.title('Fault Density Analysis - Karnataka & Andhra Pradesh')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True, alpha=0.3)
plt.savefig("visualizations/fault_density_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print("Fault density heatmap generated successfully.")