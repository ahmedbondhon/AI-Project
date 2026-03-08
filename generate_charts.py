import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# 1. Create the visualizations folder if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

print("Loading data and model...")
# Load data and model
raw_df = pd.read_csv('data/raw_data.csv')
with open('models/xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

numeric_features = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']

# Scale the data (just like in your app)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(raw_df[numeric_features])

# ==========================================
# CHART 1: Feature Importance
# ==========================================
print("Generating Feature Importance...")
plt.figure(figsize=(8, 5))
# Extract importance from XGBoost
importance = model.feature_importances_
sns.barplot(x=importance, y=numeric_features, palette='viridis')
plt.title('Which Assessments Impact the Final Grade Most?', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png') # <--- SAVING THE IMAGE
plt.close()

# ==========================================
# CHART 2: Correlation Heatmap
# ==========================================
print("Generating Correlation Heatmap...")
plt.figure(figsize=(8, 6))
correlation_matrix = raw_df[numeric_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1)
plt.title('Correlation Between Assessments', fontsize=14)
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png') # <--- SAVING THE IMAGE
plt.close()

# ==========================================
# CHART 3: Confusion Matrix
# ==========================================
print("Generating Confusion Matrix...")
plt.figure(figsize=(8, 6))

# Predict on the whole dataset to see how well it did
predictions = model.predict(scaled_features)

# NOTE: Change 'Final_Grade' if your CSV column is named differently!
actual_labels = le.transform(raw_df['Final_Grade']) 

cm = confusion_matrix(actual_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Prediction Accuracy (Confusion Matrix)', fontsize=14)
plt.xlabel('Predicted Grade')
plt.ylabel('Actual Grade')
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix.png') # <--- SAVING THE IMAGE
plt.close()

print("All charts saved successfully in the 'visualizations' folder!")