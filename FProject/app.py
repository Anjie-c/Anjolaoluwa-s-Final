import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
import base64
from PIL import Image
from io import BytesIO

img = Image.open('assets.jpg')
img = Image.open('Dashboard project.png')

# Page setup
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon=img,
    layout="wide"
)

# Custom CSS for sidebar navigation
st.markdown("""
    <style>
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a2a4f;
    }

    /* Navigation buttons */
    div[data-testid="stButton"] button {
        background-color: transparent;
        color: white;
        border: 2px solid #2a3a6f;
        border-radius: 8px;
        padding: 10px 15px;
        font-weight: 500;
        text-align: left ;
        transition: all 0.3s ease;
        width: 100%;
        justify-content: flex-start;
    }

    /* Hover effect */
    div[data-testid="stButton"] button:hover {
        background-color: #2a3a6f;
        color: white;
        border-color: #4a6a9f;
        transform: translateX(5px);
    }

    /* Active state - Navy blue background, white text, colored border */
    div[data-testid="stButton"] button:active,
    div[data-testid="stButton"] button[data-active="true"] {
        background-color: #2a3a6f;
        color: white;
        border-color: #4a6a9f;
        border-width: 2px;
        font-weight: 600;
    }

    /* Specific styling for active Predict button */
    .predict-active button {
        background-color: white;
        color: #1a2a4f;
        border-color: #1a2a4f;
        border-width: 2px;
        font-weight: 600;
    }

    .predict-active button:hover {
        background-color: #f0f2f6;
        color: #1a2a4f;
        border-color: #1a2a4f;
    }

    /* Divider */
    hr {
        border-color: #2a3a6f;
        opacity: 0.5;
    }

    /* Medical disclaimer */
    .disclaimer {
        color: #8a9aaf;
        font-size: 0.7rem;
        text-align: center;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Initializing session state
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'history_file' not in st.session_state:
    st.session_state.history_file = 'prediction_history.json'


# Save history to file
def save_history():
    try:
        with open(st.session_state.history_file, 'w') as f:
            json.dump(st.session_state.prediction_history[-100:], f)
    except:
        pass


# Load history from file
def load_history():
    if os.path.exists(st.session_state.history_file):
        try:
            with open(st.session_state.history_file, 'r') as f:
                st.session_state.prediction_history = json.load(f)
        except:
            st.session_state.prediction_history = []


# Add to history function
def add_to_history(patient_data, prediction, probability):
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'age': patient_data['Age'],
        'bmi': patient_data['BMI'],
        'fbg': patient_data.get('FBG', patient_data.get('FBG', 0)),
        'hba1c': patient_data.get('HbA1c', patient_data.get('HbA1c', 0)),
        'parent_history': patient_data['ParentalDiabetesHistory'],
        'prediction': int(prediction),
        'probability': float(probability)
    }
    st.session_state.prediction_history.insert(0, record)
    save_history()


# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_shared_Diabetes project.csv')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# Load model - Updated to load all models
@st.cache_resource
def load_models():
    try:
        stacking_model = joblib.load('models/stacking_model.pkl')
        lr_model = joblib.load('models/lr_model.pkl')
        dt_model = joblib.load('models/dt_model.pkl')
        rf_model = joblib.load('models/rf_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return stacking_model, lr_model, dt_model, rf_model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please run 'python train.py' first to train the model.")
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None, None, None


# Load only stacking model for predictions
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/stacking_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return model, scaler, feature_names
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.info("Please run 'python train.py' first to train the model.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


# Sidebar navigation - Updated with new styling
def sidebar():
    st.sidebar.markdown("Navigation")
    st.sidebar.markdown("---")

    # Dashboard button
    if st.sidebar.button("Dashboard", use_container_width=True):
        st.session_state.page = 'dashboard'
        st.rerun()

    # Predict button
    if st.sidebar.button("Single Prediction", use_container_width=True):
        st.session_state.page = 'predict'
        st.rerun()

    # Upload button
    if st.sidebar.button("Upload Local File", use_container_width=True):
        st.session_state.page = 'Upload Local File'
        st.rerun()

    # Results button
    if st.sidebar.button("Results", use_container_width=True):
        st.session_state.page = 'results'
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("Medical Disclaimer: For clinical support only")


# DASHBOARD PAGE
def dashboard_page():
    st.markdown("Diabetes Data Dashboard")

    df = load_data()
    if df is not None:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            if 'DiabetesType' in df.columns:
                diabetic_count = (df['DiabetesType'] == 1).sum()
                st.metric("Diabetes Cases", diabetic_count)
        with col3:
            if 'DiabetesType' in df.columns:
                diabetes_rate = (df['DiabetesType'].mean() * 100)
                st.metric("Diabetes Rate", f"{diabetes_rate:.1f}%")
        with col4:
            if 'ResidentUrbanRural' in df.columns:
                urban_count = (df['ResidentUrbanRural'] == 2).sum()  # No quotes around 1
                st.metric("Urban Patients", urban_count)

        st.markdown("---")

        # Check for model metrics graphs
        st.markdown("Model Evaluation Metrics")
        metrics_files = [
            'confusion_matrix.png',
            'roc_curve.png',
            'pr_curve.png',
            'metrics_comparison.png',
            'model_accuracy_comparison.png',
            'feature_importance.png',
            'cv_scores.png'
        ]

        metrics_exist = all([os.path.exists(f'models/{f}') for f in metrics_files])

        if metrics_exist:
            # Row 1: Confusion Matrix + Metrics Comparison
            col1, col2 = st.columns(2)
            with col1:
                st.image('models/confusion_matrix.png', use_container_width=True, caption='Confusion Matrix')
            with col2:
                st.image('models/metrics_comparison.png', use_container_width=True, caption='Performance Metrics')

            # Row 2: ROC Curve + PR Curve
            col1, col2 = st.columns(2)
            with col1:
                st.image('models/roc_curve.png', use_container_width=True, caption='ROC Curve')
            with col2:
                st.image('models/pr_curve.png', use_container_width=True, caption='Precision-Recall Curve')

            # Row 3: Accuracy Comparison + Feature Importance
            col1, col2 = st.columns(2)
            with col1:
                st.image('models/model_accuracy_comparison.png', use_container_width=True,
                         caption='Model Accuracy Comparison')
            with col2:
                st.image('models/feature_importance.png', use_container_width=True, caption='Feature Importance')

            # Row 4: CV Scores + Classification Report
            col1, col2 = st.columns(2)
            with col1:
                st.image('models/cv_scores.png', use_container_width=True, caption='Cross-Validation Scores')
            with col2:
                if os.path.exists('models/classification_report_heatmap.png'):
                    st.image('models/classification_report_heatmap.png', use_container_width=True,
                             caption='Classification Report')
        else:
            st.info("Run 'python train.py' to generate evaluation metrics graphs.")

        st.markdown("---")

        # Geographic Clustering
        st.markdown("Geographic View of Diabetes")

        if 'ResidentUrbanRural' in df.columns and 'DiabetesType' in df.columns:
            col1, col2 = st.columns(2)

            with col1:
                location_counts = df['ResidentUrbanRural'].value_counts()
                fig = px.pie(
                    values=location_counts.values,
                    names=location_counts.index,
                    title='Patient Distribution by Residence',
                    color_discrete_sequence=['#2C3E50', '#E74C3C']
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                location_diabetes = df.groupby('ResidentUrbanRural')['DiabetesType'].mean() * 100
                fig = px.bar(
                    x=location_diabetes.index,
                    y=location_diabetes.values,
                    title='Diabetes Rate by Location',
                    labels={'x': 'Location', 'y': 'Diabetes Rate (%)'},
                    color=location_diabetes.values,
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            # Scatter plot
            st.markdown("Patient Clusters by Location")
            fig = px.scatter(
                df,
                x='Age',
                y='BMI',
                color='ResidentUrbanRural',
                symbol='DiabetesType',
                title='Patient Clusters: Age vs BMI by Location',
                color_discrete_sequence=['#2C3E50', '#E74C3C'],
                hover_data=['FBG', 'HbA1c']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

        # charts
        st.markdown("---")
        st.markdown("Data Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            if 'Age' in df.columns:
                fig = px.histogram(df, x='Age', title='Age Distribution', color_discrete_sequence=['#2C3E50'])
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            if 'BMI' in df.columns and 'DiabetesType' in df.columns:
                fig = px.box(df, x='DiabetesType', y='BMI', title='BMI by Diabetes Status',
                             color='DiabetesType', color_discrete_sequence=['#27AE60', '#E74C3C'])
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.subheader("Power BI Dashboard")

        st.image(
            "Dashboard project.png",
            caption="Diabetes Analytics Dashboard",
            use_container_width=True
        )

        with st.expander("View Raw Data"):
            st.dataframe(df, use_container_width=True)


# SINGLE PREDICTION PAGE
def predict_page():
    st.markdown("Diabetes Risk Prediction")

    model, scaler, feature_names = load_model()
    if model is None:
        st.error("Model not found. Please run 'python train.py' first")
        st.info("Make sure the file 'models/stacking_model.pkl' exists.")
        return

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("Patient Features")

        age = st.number_input(" Age (years)", min_value=0, max_value=120, value=45, step=1)

        st.markdown("Body Measurements")

        # BMI Calculation - Weight and Height inputs
        weight = st.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5)
        height = st.number_input("Height (m)", min_value=1.0, max_value=2.5, value=1.75, step=0.01)

        # Calculate BMI automatically
        if height > 0:
            bmi = weight / (height ** 2)
        else:
            bmi = 0

        # Display calculated BMI
        st.markdown(f"""
            <div style="background: #1a2a4f; padding: 9px; border-radius: 8px; text-align: center; margin: 10px 0;">
                <span style="font-size: 16px; font-weight: bold;"> Calculated BMI: </span>
                <span style="font-size: 20px; font-weight: bold; color: white;">{bmi:.1f}</span>
            </div>
        """, unsafe_allow_html=True)

        # BMI Category indicator
        if bmi < 18.5:
            st.caption("Underweight")
        elif bmi < 25:
            st.caption("Normal weight")
        elif bmi < 30:
            st.caption("Overweight")
        else:
            st.caption("Obese")

    with col2:
        st.markdown("Blood Glucose Levels")

        fbg = st.number_input("Fasting Blood Glucose (FBG) (mg/dL)",
                              min_value=50, max_value=500, value=100, step=5,
                              help="Normal: 70-99 mg/dL | Prediabetes: 100-125 mg/dL | Diabetes: 126+ mg/dL")

        hba1c = st.number_input("HbA1c Level (%)",
                                min_value=4.0, max_value=15.0, value=5.5, step=0.1,
                                help="Normal: <5.7% | Prediabetes: 5.7-6.4% | Diabetes: 6.5%+")

        st.markdown("Family History")

        parent_history = st.selectbox("Parental Diabetes History", ["No", "Yes"],
                                      help="Does either parent or family have diabetes?")
        parent_value = 1 if parent_history == "Yes" else 0

        # Show risk indication for family history
        if parent_value == 1:
            st.caption("Family history of diabetes might increase risk")
        else:
            st.caption("No family history of diabetes")

    if st.button("Predict", type="primary"):
        # Create input with correct column names
        input_data = pd.DataFrame([{
            'FBG': fbg,
            'HbA1c': hba1c,
            'BMI': bmi,
            'Age': age,
            'ParentalDiabetesHistory': parent_value
        }])

        # Add engineered features
        input_data['BMI_Age'] = input_data['BMI'] * input_data['Age'] / 100
        input_data['Glucose_HbA1c'] = input_data['FBG'] * input_data['HbA1c'] / 100
        input_data['Metabolic_Risk'] = (input_data['FBG'] / 100 +
                                        input_data['HbA1c'] / 10 +
                                        input_data['BMI'] / 30) / 3
        input_data['Log_FBG'] = np.log1p(input_data['FBG'])
        input_data['Log_HbA1c'] = np.log1p(input_data['HbA1c'])

        # Ensure all feature columns exist
        if feature_names:
            for col in feature_names:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_names]

        # Scale and predict
        X_scaled = scaler.transform(input_data)
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]

        # Save to history
        patient_data = {
            'Age': age,
            'BMI': bmi,
            'FBG': fbg,
            'HbA1c': hba1c,
            'ParentalDiabetesHistory': parent_value
        }
        add_to_history(patient_data, prediction, probability)

        st.markdown("---")
        st.markdown("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error(f"HIGH RISK - {probability:.1%} probability of diabetes")
                st.warning("Please consult a healthcare provider immediately.")
            else:
                st.success(f"LOW RISK - {(1 - probability):.1%} probability of no diabetes")

        with col2:
            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                title={'text': "Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#E74C3C" if prediction == 1 else "#27AE60"},
                    'steps': [
                        {'range': [0, 30], 'color': "#2ECC71"},
                        {'range': [30, 70], 'color': "#F1C40F"},
                        {'range': [70, 100], 'color': "#E74C3C"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        # Show detailed risk factors
        st.markdown("Risk Factor Analysis")

        # Display which factors are contributing to risk
        risk_factors = []

        if fbg >= 126:
            risk_factors.append("High Fasting Blood Glucose (>126 mg/dL)")
        elif fbg >= 100:
            risk_factors.append("Prediabetes range (100-125 mg/dL)")

        if hba1c >= 6.5:
            risk_factors.append("High HbA1c (>6.5%)")
        elif hba1c >= 5.7:
            risk_factors.append("Prediabetes range (5.7-6.4%)")

        if bmi >= 30:
            risk_factors.append("Obese (BMI ≥ 30)")
        elif bmi >= 25:
            risk_factors.append("Overweight (BMI 25-29.9)")

        if age >= 45:
            risk_factors.append("Age ≥ 45 (increased risk)")

        if parent_value == 1:
            risk_factors.append("Family history of diabetes")

        if risk_factors:
            st.markdown(" Risk Factors Identified:**")
            for factor in risk_factors:
                st.write(f"- {factor}")
        else:
            st.success(" No significant risk factors identified")

def Upload_Local_File_page():
    st.markdown("Batch Patient Upload")

    model, scaler, feature_names = load_model()
    if model is None:
        st.error("Model not found. Please run 'python train.py' first")
        st.info("Make sure the file 'models/stacking_model.pkl' exists.")
        return

    # Custom CSS
    st.markdown("""
        <style>
        div[data-testid="stFileUploader"] {
            background: transparent;
            border: none;
            padding: 0;
            width: 100%;
        }
        div[data-testid="stFileUploader"] > div {
            background: transparent;
            border: none;
            padding: 0;
            justify-content: center;
            align-items: center;
        }
        div[data-testid="stFileUploader"] button {
            background: linear-gradient(135deg, #1a2a4f 0%, #2a3a6f 100%);
            color: white;
            border-radius: 12px;
            padding: 15px 30px;
            font-size: 16px;
            font-weight: bold;
            justify-content: center;
            width: 100%;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 15px;
        }
        div[data-testid="stFileUploader"] button:hover {
            background: linear-gradient(135deg, #2a3a6f 0%, #3a4a7f 100%);
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        </style>
    """, unsafe_allow_html=True)

    # upload
    uploaded_file = st.file_uploader("", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.markdown("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total records: {len(df)}")


        # CASE-INSENSITIVE

        column_mapping = {
            'FBG': ['fbg', 'FastingBloodGlucose', 'Glucose', 'Fasting_Glucose', 'FBG_Level','Blood glucose', 'BLOOD GLUCOSE'],
            'HbA1c': ['hba1c', 'Hba1c', 'HbA1c_Level', 'Hemoglobin_A1c', 'A1C', 'HBA1C'],
            'BMI': ['bmi', 'BodyMassIndex', 'Bmi', 'BMI_Value'],
            'Age': ['age', 'Age_Years', 'PatientAge'],
            'ParentalDiabetesHistory': ['parentalhistory', 'Parental_History', 'FamilyHistory',
                                        'Family_History', 'ParentDiabetes', 'ParentalDiabetes', 'diabetes_history']
        }

        # Create a mapping from actual column names to expected names
        actual_columns = {}
        for expected, variations in column_mapping.items():
            for col in df.columns:
                col_lower = col.lower().strip()
                if col_lower in [v.lower() for v in variations] or col_lower == expected.lower():
                    actual_columns[expected] = col
                    break

        # Check if all required columns were found
        required_cols = ['FBG', 'HbA1c', 'BMI', 'Age', 'ParentalDiabetesHistory']
        missing_cols = [col for col in required_cols if col not in actual_columns]

        if missing_cols:
            st.error(f"Could not find these required columns: {missing_cols}")
            st.info(f"Please ensure your CSV has columns like: {required_cols}")
            st.write("Columns found in your file:", df.columns.tolist())
            return

        # Rename columns to match expected names
        rename_dict = {actual_columns[col]: col for col in required_cols}
        df = df.rename(columns=rename_dict)

        # Show the mapping for transparency
        st.success(f"Successfully mapped columns: {list(rename_dict.values())} → {list(rename_dict.keys())}")

        # Now proceed with prediction using the standardized column names
        if st.button("Predict", type="primary", use_container_width=True):
            with st.spinner("Processing predictions..."):
                X = df[required_cols].copy()
                X = X.fillna(X.median())

                # Add engineered features
                X['BMI_Age'] = X['BMI'] * X['Age'] / 100
                X['Glucose_HbA1c'] = X['FBG'] * X['HbA1c'] / 100
                X['Metabolic_Risk'] = (X['FBG'] / 100 +
                                       X['HbA1c'] / 10 +
                                       X['BMI'] / 30) / 3
                X['Log_FBG'] = np.log1p(X['FBG'])
                X['Log_HbA1c'] = np.log1p(X['HbA1c'])

                if feature_names:
                    for col in feature_names:
                        if col not in X.columns:
                            X[col] = 0
                    X = X[feature_names]

                X_scaled = scaler.transform(X)

                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]

                df['Risk_Level'] = ['High Risk' if p == 1 else 'Low Risk' for p in predictions]
                df['Risk_Score'] = [f"{p:.1%}" for p in probabilities]

                high_risk_count = (predictions == 1).sum()

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Patients", len(df))
                with col2:
                    st.metric("High Risk", high_risk_count)
                with col3:
                    st.metric("Low Risk", (predictions == 0).sum())

                if high_risk_count > 0:
                    st.error(f"ALERT: {high_risk_count} patient(s) identified as HIGH RISK")

                st.markdown("Prediction Results")
                st.dataframe(df[required_cols + ['Risk_Level', 'Risk_Score']], use_container_width=True)

                for idx, row in df.iterrows():
                    patient_data = {
                        'Age': row['Age'],
                        'BMI': row['BMI'],
                        'FBG': row['FBG'],
                        'HbA1c': row['HbA1c'],
                        'ParentalDiabetesHistory': row['ParentalDiabetesHistory']
                    }
                    add_to_history(patient_data, predictions[idx], probabilities[idx])

                csv = df.to_csv(index=False)
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "prediction_results.csv",
                    "text/csv",
                    use_container_width=True
                )

                st.success("Predictions complete! Results saved to history.")

    else:
        st.info("Click 'Browse Files' above to upload your CSV file")


# RESULTS PAGE
def results_page():
    st.markdown("My Prediction Results")

    if len(st.session_state.prediction_history) > 0:
        df_results = pd.DataFrame(st.session_state.prediction_history)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Assessments", len(df_results))
        with col2:
            high_risk = (df_results['prediction'] == 1).sum()
            st.metric("High Risk Cases", high_risk)
        with col3:
            risk_rate = (high_risk / len(df_results)) * 100
            st.metric("Risk Rate", f"{risk_rate:.1f}%")

        st.dataframe(df_results, use_container_width=True)

        csv = df_results.to_csv(index=False)
        st.download_button("Export History", csv, "prediction_history.csv", "text/csv")
    else:
        st.info("No predictions yet. Go to the Predict page to get started.")


# MAIN
def main():
    load_history()
    sidebar()

    if st.session_state.page == 'dashboard':
        dashboard_page()
    elif st.session_state.page == 'predict':
        predict_page()
    elif st.session_state.page == 'Upload Local File':
        Upload_Local_File_page()
    elif st.session_state.page == 'results':
        results_page()


if __name__ == "__main__":
    main()