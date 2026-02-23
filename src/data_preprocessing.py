import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import pickle

def preprocess_data():
    print("Loading DIU raw data...")
    df = pd.read_csv('data/raw_data.csv')

    # 1. Encode the categorical target (Grade)
    print("Encoding DIU Grades...")
    le = LabelEncoder()
    df['Grade_Encoded'] = le.fit_transform(df['Grade'])
    
    # Save the label encoder so the app can convert numbers back to 'A+', 'B', etc.
    os.makedirs('models', exist_ok=True)
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)

    # 2. Normalize numeric data
    print("Normalizing assessment marks...")
    numeric_features = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']
    
    scaler = StandardScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Drop the original text grade column, keep only the encoded one for the AI
    df = df.drop(columns=['Grade'])

    # 3. Save processed data
    output_path = 'data/processed_data.csv'
    df.to_csv(output_path, index=False)
    print("Data preprocessing complete! Ready for multi-class training.")

if __name__ == "__main__":
    preprocess_data()