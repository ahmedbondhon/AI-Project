import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="DIU Grade Predictor Pro", layout="wide", page_icon="21")

# --- 1. Load Model, Scaler, and Label Encoder ---
@st.cache_resource
def load_setup():
    # Load the new XGBoost brain!
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
        
    raw_df = pd.read_csv('data/raw_data.csv')
    numeric_features = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']
    scaler = StandardScaler()
    scaler.fit(raw_df[numeric_features])
    
    return model, scaler, le

model, scaler, le = load_setup()

# --- 2. Sidebar Navigation ---
st.sidebar.image("https://daffodilvarsity.edu.bd/images/logo.png", width=200)
st.sidebar.title("DIU Assessment AI (XGBoost + SHAP)")
page = st.sidebar.radio("Navigation:", ["Single Grade Predictor", "Bulk Class Predictor", "Analytics Dashboard"])

st.sidebar.divider()
st.sidebar.info("**Course:** Artificial Intelligence\n\n**Student:** Nazmul Ahmed Bondhon (19-A)\n\n**Supervisor:** Dr. Md. Mizanur Rahoman")

# ==========================================
# PAGE 1: SINGLE PREDICTOR & SHAP EXPLANATION
# ==========================================
if page == "Single Grade Predictor":
    st.title("DIU Pre-Final Grade Predictor (Explainable AI)")
    st.write("Enter the student's continuous assessment marks (Total 60) to predict their Final Letter Grade.")

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.number_input("Attendance (Out of 7)", min_value=0, max_value=7, value=6)
        class_test = st.number_input("Class Test CT1, CT2, CT3 (Out of 15)", min_value=0, max_value=15, value=12)
        assignment = st.number_input("Assignment (Out of 5)", min_value=0, max_value=5, value=4)
        
    with col2:
        presentation = st.number_input("Presentation (Out of 8)", min_value=0, max_value=8, value=7)
        midterm = st.number_input("Midterm Examination (Out of 25)", min_value=0, max_value=25, value=20)

    total_current = attendance + class_test + assignment + presentation + midterm
    st.info(f"**Current Marks Before Final:** {total_current} / 60")

    if st.button("Predict & Explain Grade", type="primary"):
        # Prepare Data
        feature_names = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']
        raw_input = pd.DataFrame([[attendance, class_test, assignment, presentation, midterm]], columns=feature_names)
        
        # Scale inputs
        scaled_input = scaler.transform(raw_input)
        scaled_df = pd.DataFrame(scaled_input, columns=feature_names)
        
        # Predict
        prediction_encoded = model.predict(scaled_df)[0]
        predicted_grade = le.inverse_transform([prediction_encoded])[0]
        
        st.divider()
        st.success(f"###AI Prediction: Based on current performance, the expected Final Grade is: **{predicted_grade}**")
        
        # --- SHAP EXPLAINABLE AI ---
        st.markdown("###Why did the AI choose this grade?")
        st.write("This **SHAP Waterfall Plot** shows exactly which marks pushed the prediction towards this specific grade, and which marks pulled it away.")
        
        try:
            # Generate SHAP values
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(scaled_df)
            
            # Extract values for the specifically predicted class
            shap_val_single = shap.Explanation(
                values=shap_values.values[0, :, prediction_encoded],
                base_values=shap_values.base_values[0, prediction_encoded],
                data=raw_input.iloc[0].values, # Show actual marks on the chart, not scaled numbers
                feature_names=feature_names
            )
            
            # Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_val_single, show=False)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Could not generate SHAP explanation. Error: {e}")

# ==========================================
# PAGE 2: BULK CLASS PREDICTOR 
# ==========================================
elif page == "Bulk Class Predictor":
    st.title("Bulk Class Grade Prediction")
    st.write("Upload a CSV file containing your entire class roster's continuous assessment marks to instantly predict everyone's Final Letter Grade.")
    
    uploaded_file = st.file_uploader("Upload Class CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            bulk_df = pd.read_csv(uploaded_file)
            if st.button("Generate Class Predictions", type="primary"):
                process_df = bulk_df.copy()
                features = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']
                process_df[features] = scaler.transform(process_df[features])
                predictions_encoded = model.predict(process_df[features])
                bulk_df['Predicted_Final_Grade'] = le.inverse_transform(predictions_encoded)
                st.success("Predictions generated successfully!")
                st.dataframe(bulk_df)
                
                csv = bulk_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Results as CSV", data=csv, file_name='diu_class_predictions.csv', mime='text/csv')
        except Exception as e:
            st.error(f"Error processing file. Details: {e}")

# ==========================================
# PAGE 3: DYNAMIC ANALYTICS DASHBOARD
# ==========================================
elif page == "Analytics Dashboard":
    st.title("DIU Assessment Analytics")
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Correlation", "Prediction Accuracy"])
    
    # We need to load the raw data here to draw the graphs
    raw_df = pd.read_csv('data/raw_data.csv')
    numeric_features = ['Attendance', 'Class_Test', 'Assignment', 'Presentation', 'Midterm']
    
    # --- TAB 1: Feature Importance ---
    with tab1:
        st.subheader("Which Assessments Impact the Final Grade Most?")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        importance = model.feature_importances_
        sns.barplot(x=importance, y=numeric_features, palette='viridis', ax=ax1)
        ax1.set_xlabel('Importance Score')
        st.pyplot(fig1)

    # --- TAB 2: Correlation Heatmap ---
    with tab2:
        st.subheader("Correlation Between Assessments")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        correlation_matrix = raw_df[numeric_features].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, ax=ax2)
        st.pyplot(fig2)

    # --- TAB 3: Confusion Matrix ---
    with tab3:
        st.subheader("Prediction Accuracy (Confusion Matrix)")
        
        # AUTO-DETECT TARGET COLUMN: Looks for common grade column names, or defaults to the last column
        possible_targets = ['Final_Grade', 'Grade', 'Result', 'Target', 'Class']
        target_column = next((col for col in possible_targets if col in raw_df.columns), raw_df.columns[-1])
        
        try:
            fig3, ax3 = plt.subplots(figsize=(8, 6))
            scaled_features = scaler.transform(raw_df[numeric_features])
            predictions = model.predict(scaled_features)
            actual_labels = le.transform(raw_df[target_column])
            
            cm = confusion_matrix(actual_labels, predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=le.classes_, yticklabels=le.classes_, ax=ax3)
            ax3.set_xlabel('Predicted Grade')
            ax3.set_ylabel(f'Actual Grade (Column: {target_column})')
            st.pyplot(fig3)
        except Exception as e:
            st.error(f"Could not generate Confusion Matrix. The auto-detected target column was '{target_column}'. Ensure your target labels match what the model was trained on. Error: {e}")