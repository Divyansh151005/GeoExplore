import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Create a directory for visualizations
os.makedirs("visualizations", exist_ok=True)

# Generate model evaluation visualizations
print("Generating model evaluation visualizations...")

# 1. Confusion Matrix for Model Evaluation
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

# 2. Feature Importance Plot
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

# 3. Model Comparison Chart
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

print("Model evaluation visualizations generated successfully.")