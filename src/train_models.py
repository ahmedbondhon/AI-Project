import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score
import pickle
import os

def train_and_evaluate():
    os.makedirs('models', exist_ok=True)
    df = pd.read_csv('data/processed_data.csv')

    X = df.drop(columns=['Grade_Encoded'])
    y = df['Grade_Encoded']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Basic Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    print("\n--- Training Traditional Models ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"{name} -> Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    # 2. State-of-the-Art Model (XGBoost)
    print("\n--- Training Advanced XGBoost Classifier ---")
    # XGBoost handles complex patterns much better than traditional trees
    xgb_model = XGBClassifier(
        n_estimators=150, 
        learning_rate=0.1, 
        max_depth=6, 
        random_state=42,
        eval_metric='mlogloss'
    )
    
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    print(f" XGBoost Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
    
    # Save the new XGBoost model as our primary brain
    with open("models/xgboost_model.pkl", 'wb') as f:
        pickle.dump(xgb_model, f)

    print("\nAll models trained! XGBoost is saved and ready.")

if __name__ == "__main__":
    train_and_evaluate()