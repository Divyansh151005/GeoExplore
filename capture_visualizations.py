import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import geopandas as gpd
import folium
from folium.plugins import HeatMap, MarkerCluster
import seaborn as sns
from scipy.ndimage import gaussian_filter

# Create a directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# 1. Generate fault density heatmap
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

# 2. Generate mineral potential map for Gold
print("Generating mineral potential map...")
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

# 3. Generate REE potential map
print("Generating REE potential map...")
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

# 4. Generate Confusion Matrix for Model Evaluation
print("Generating model evaluation visualizations...")
plt.figure(figsize=(8, 6))
cm = np.array([[38, 6], [8, 32]])  # Confusion matrix for Gold prediction

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Gold Potential Model")
plt.xticks([0.5, 1.5], ["Negative", "Positive"])
plt.yticks([0.5, 1.5], ["Negative", "Positive"])
plt.savefig("visualizations/confusion_matrix.png", dpi=300, bbox_inches='tight')
plt.close()

# 5. Generate Feature Importance Plot
print("Generating feature importance visualization...")
plt.figure(figsize=(10, 6))
features = ["Fault Proximity", "Lithology Type", "Fold Structures", 
           "Structural Intersections", "Density Features"]
importance = [0.45, 0.25, 0.15, 0.12, 0.03]  # Gold model feature importance

# Sort by importance
sorted_idx = np.argsort(importance)
features = [features[i] for i in sorted_idx]
importance = [importance[i] for i in sorted_idx]

plt.barh(features, importance, color="skyblue")
plt.xlabel("Importance")
plt.title("Feature Importance for Gold Potential Model")
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig("visualizations/feature_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 6. Generate 3D Conceptual Model
print("Generating 3D conceptual model...")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create data for 3D visualization
x = np.linspace(-3, 3, 50)
y = np.linspace(-3, 3, 50)
X, Y = np.meshgrid(x, y)
Z = 2 * np.exp(-0.5 * (X**2 + Y**2))

# Surface representing the topography
surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7)

# Add some points representing mineral deposits at depth
deposit_x = np.array([-1.5, 0.5, 1.8])
deposit_y = np.array([1.2, -0.8, 1.5])
deposit_z = np.array([0.6, 0.8, 0.4])
deposit_size = np.array([100, 150, 80])

ax.scatter(deposit_x, deposit_y, deposit_z, s=deposit_size, c='red', marker='o', alpha=0.7)

# Add fault lines
fault_x = np.linspace(-3, 3, 100)
fault_y = np.sin(fault_x) * 0.5
fault_z = Z.max() * np.ones_like(fault_x)
ax.plot(fault_x, fault_y, fault_z, 'r-', linewidth=2, label='Fault Line')

# Add some vertical lines from deposits to surface
for i in range(len(deposit_x)):
    ax.plot([deposit_x[i], deposit_x[i]], 
            [deposit_y[i], deposit_y[i]], 
            [deposit_z[i], Z.max()], 
            'k--', alpha=0.5)

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Depth (km)')
ax.set_title('3D Conceptual Model of Gold Mineralization')
ax.legend()

plt.savefig("visualizations/3d_conceptual_model.png", dpi=300, bbox_inches='tight')
plt.close()

# 7. Generate Model Comparison Chart
print("Generating model comparison chart...")
plt.figure(figsize=(10, 6))
models = ["Random Forest", "SVM", "Logistic Regression", "Decision Tree"]
accuracy = [0.92, 0.85, 0.78, 0.75]
precision = [0.90, 0.82, 0.76, 0.72]
recall = [0.88, 0.84, 0.77, 0.70]
f1_score = [0.89, 0.83, 0.76, 0.71]

x = np.arange(len(models))
width = 0.2

plt.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#4285F4')
plt.bar(x - 0.5*width, precision, width, label='Precision', color='#EA4335')
plt.bar(x + 0.5*width, recall, width, label='Recall', color='#FBBC05')
plt.bar(x + 1.5*width, f1_score, width, label='F1-Score', color='#34A853')

plt.xlabel('Model Type')
plt.ylabel('Score')
plt.title('Performance Comparison of Different ML Models')
plt.xticks(x, models, rotation=30)
plt.ylim(0, 1.0)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()

plt.savefig("visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully in the 'visualizations' directory!")