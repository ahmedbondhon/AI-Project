import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import os

def generate_visualizations():
    os.makedirs('visualizations', exist_ok=True)
    
    print("Loading DIU data and models...")
    df = pd.read_csv('data/processed_data.csv')
    
    with open('models/xgboost_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
        
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    X = df.drop(columns=['Grade_Encoded'])
    y = df['Grade_Encoded']

    # 1. Correlation Heatmap
    print("Generating Correlation Heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of DIU Assessment Tasks')
    plt.tight_layout()
    plt.savefig('visualizations/correlation_heatmap.png')
    plt.close()

    # 2. Feature Importance Chart
    print("Generating Feature Importance Chart...")
    importances = rf_model.feature_importances_
    features = ['Attendance (7)', 'Class Test (15)', 'Assignment (5)', 'Presentation (8)', 'Midterm (25)']
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features, hue=features, palette='viridis', legend=False)
    plt.title('Which Task Impacts the Final Grade the Most?')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

    # 3. Confusion Matrix
    print("Generating Confusion Matrix...")
    y_pred = rf_model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    # Get the actual grade labels (A+, A, etc.) in the correct order
    labels = le.inverse_transform(range(len(le.classes_)))
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(cmap='Blues', ax=ax, xticks_rotation=45)
    plt.title('Confusion Matrix (DIU Grade Prediction)')
    plt.tight_layout()
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

    print("\nSuccess! Visualizations saved.")

if __name__ == "__main__":
    generate_visualizations()