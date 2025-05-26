import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import warnings
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import json
warnings.filterwarnings('ignore')

# Set page config with custom theme
st.set_page_config(
    page_title="Obesity Prediction App",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-size: 1.2em;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: scale(1.02);
    }
    .css-1d391kg {
        padding: 1rem;
    }
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stSuccess {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and label encoder
@st.cache_resource
def load_model():
    try:
        # Check if files exist
        model_files = ['xgb_model.json', 'xgb_model.bin', 'xgb_model_new.pkl']
        model_file = None
        
        for file in model_files:
            if os.path.exists(file):
                model_file = file
                break
                
        if model_file is None:
            st.error("No model file found!")
            return None, None
            
        if not os.path.exists('label_encoder.joblib'):
            st.error("Label encoder file 'label_encoder.joblib' not found!")
            return None, None

        # Load model
        try:
            if model_file.endswith('.json') or model_file.endswith('.bin'):
                model = xgb.Booster()
                model.load_model(model_file)
            else:  # .pkl
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None, None
        
        # Load label encoder
        try:
            label_encoder = joblib.load('label_encoder.joblib')
        except Exception as e:
            st.error(f"Error loading label encoder: {str(e)}")
            return None, None
            
        return model, label_encoder
    except Exception as e:
        st.error(f"Error in model loading process: {str(e)}")
        return None, None

# Main app function
def main():
    # Title with animation
    st.markdown("""
        <div style='text-align: center; padding: 1rem;'>
            <h1 style='color: #2E4053; font-size: 2.5em; margin-bottom: 0.5em;'>
                🏥 Obesity Prediction App
            </h1>
            <p style='color: #566573; font-size: 1.2em;'>
                Enter your information to predict the obesity level
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs for different sections
    tab1, tab2 = st.tabs(["📝 Input Form", "📊 Results & Analysis"])
    
    with tab1:
        # Create input fields in two columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
            gender = st.selectbox("Gender", ["Female", "Male"])
            height = st.number_input("Height (cm)", min_value=0.0, max_value=300.0, value=170.0)
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0)
            family_history = st.selectbox("Family History of Obesity", ["Yes", "No"])
            
            st.markdown("### Eating Habits")
            favc = st.selectbox("Frequent High Caloric Food", ["Yes", "No"])
            fcvc = st.number_input("Vegetable Consumption (1-3)", min_value=1, max_value=3, value=2)
            ncp = st.number_input("Number of Main Meals", min_value=1, max_value=5, value=3)
            caec = st.selectbox("Food Between Meals", ["Sometimes", "Frequently", "Always", "Never"])
            
        with col2:
            st.markdown("### Physical Activity")
            smoke = st.selectbox("Smoking", ["Yes", "No"])
            ch2o = st.number_input("Daily Water Consumption (L)", min_value=0.0, max_value=10.0, value=2.0)
            scc = st.selectbox("Calorie Monitoring", ["Yes", "No"])
            faf = st.number_input("Physical Activity Frequency (days/week)", min_value=0, max_value=7, value=3)
            tue = st.number_input("Technology Usage (hours/day)", min_value=0, max_value=24, value=2)
            
            st.markdown("### Transportation")
            calc = st.selectbox("Transportation Mode", ["Public_Transportation", "Automobile", "Motorbike", "Bike", "Walking"])
            mtrans = st.selectbox("Transportation Time (minutes)", ["0-15", "15-30", "30-60", "60-120", ">120"])

        # Prediction button with animation
        if st.button("🔮 Predict Obesity Level", use_container_width=True):
            with st.spinner('Analyzing your data...'):
                try:
                    # Load model and encoder
                    model, label_encoder = load_model()
                    if model is None or label_encoder is None:
                        st.error("Model or label encoder not loaded.")
                        return

                    # Map categorical variables
                    gender_mapping = {"Female": 0, "Male": 1}
                    family_history_mapping = {"No": 0, "Yes": 1}
                    favc_mapping = {"No": 0, "Yes": 1}
                    caec_mapping = {"Never": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
                    smoke_mapping = {"No": 0, "Yes": 1}
                    scc_mapping = {"No": 0, "Yes": 1}
                    calc_mapping = {
                        "Public_Transportation": 0,
                        "Automobile": 1,
                        "Motorbike": 2,
                        "Bike": 3,
                        "Walking": 4
                    }
                    mtrans_mapping = {
                        "0-15": 0,
                        "15-30": 1,
                        "30-60": 2,
                        "60-120": 3,
                        ">120": 4
                    }

                    # Input as DataFrame
                    input_data = pd.DataFrame({
                        'Gender': [gender_mapping[gender]],
                        'Age': [age],
                        'Height': [height],
                        'Weight': [weight],
                        'family_history_with_overweight': [family_history_mapping[family_history]],
                        'FAVC': [favc_mapping[favc]],
                        'FCVC': [fcvc],
                        'NCP': [ncp],
                        'CAEC': [caec_mapping[caec]],
                        'SMOKE': [smoke_mapping[smoke]],
                        'CH2O': [ch2o],
                        'SCC': [scc_mapping[scc]],
                        'FAF': [faf],
                        'TUE': [tue],
                        'CALC': [calc_mapping[calc]],
                        'MTRANS': [mtrans_mapping[mtrans]]
                    })

                    dmatrix = xgb.DMatrix(input_data)
                    prediction = model.predict(dmatrix)

                    # Determine predicted class index safely
                    if isinstance(prediction, np.ndarray):
                        if prediction.ndim == 2:
                            predicted_class_idx = np.argmax(prediction[0])
                        elif prediction.ndim == 1 and prediction.shape[0] > 1:
                            predicted_class_idx = np.argmax(prediction)
                        else:
                            predicted_class_idx = int(np.round(prediction[0]))
                    else:
                        predicted_class_idx = int(np.round(float(prediction)))

                    predicted_class = label_encoder.inverse_transform([predicted_class_idx])[0]

                    # Calculate BMI
                    height_m = height / 100
                    bmi = weight / (height_m * height_m)

                    # Switch to Results tab
                    st.session_state['prediction_results'] = {
                        'predicted_class': predicted_class,
                        'bmi': bmi,
                        'model': model,
                        'input_data': input_data
                    }
                    st.rerun()

                except Exception as e:
                    st.error("An error occurred during prediction.")
                    st.write("Error type:", type(e).__name__)
                    st.write("Error message:", str(e))

    with tab2:
        if 'prediction_results' in st.session_state:
            results = st.session_state['prediction_results']
            
            # Create two columns for results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Prediction Results")
                st.success(f"Predicted Obesity Level: {results['predicted_class']}")
                
                # BMI display with animation
                st.markdown("### BMI Analysis")
                st.metric("BMI", f"{results['bmi']:.1f}", 
                         help="Body Mass Index (BMI) = Weight (kg) / Height (m)²")
                
                # BMI Category with color coding
                bmi = results['bmi']
                if bmi < 18.5:
                    bmi_category = "Underweight"
                    color = "blue"
                elif bmi < 25:
                    bmi_category = "Normal weight"
                    color = "green"
                elif bmi < 30:
                    bmi_category = "Overweight"
                    color = "orange"
                else:
                    bmi_category = "Obese"
                    color = "red"
                
                st.markdown(f"""
                    <div style='background-color: {color}; color: white; padding: 1rem; border-radius: 5px; text-align: center;'>
                        <h3>BMI Category: {bmi_category}</h3>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Create BMI gauge chart with animation
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = bmi,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 40]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 18.5], 'color': "lightgray"},
                            {'range': [18.5, 25], 'color': "green"},
                            {'range': [25, 30], 'color': "yellow"},
                            {'range': [30, 40], 'color': "red"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': bmi
                        }
                    },
                    title = {'text': "BMI Gauge"}
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance visualization with animation
            st.markdown("### Feature Importance Analysis")
            importance = results['model'].get_score(importance_type='weight')
            importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            
            # Create DataFrame for feature importance
            importance_df = pd.DataFrame({
                'Features': list(importance.keys()),
                'Importance': list(importance.values())
            })
            
            fig = px.bar(
                importance_df,
                x='Features',
                y='Importance',
                title='Feature Importance in Prediction',
                color='Importance',
                color_continuous_scale='Viridis'
            )
            
            # Update layout for better visualization
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Importance Score",
                showlegend=False,
                hovermode='x'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Health recommendations with animation
            st.markdown("### Personalized Health Recommendations")
            if bmi < 18.5:
                st.info("""
                #### For Underweight Individuals:
                - Consider consulting a nutritionist to develop a healthy weight gain plan
                - Focus on nutrient-dense foods
                - Include strength training in your exercise routine
                """)
            elif bmi < 25:
                st.success("""
                #### For Normal Weight Individuals:
                - Maintain your current healthy lifestyle
                - Continue regular physical activity
                - Keep a balanced diet
                """)
            elif bmi < 30:
                st.warning("""
                #### For Overweight Individuals:
                - Consider increasing physical activity
                - Focus on portion control
                - Include more fruits and vegetables in your diet
                """)
            else:
                st.error("""
                #### For Obese Individuals:
                - Consult with a healthcare provider
                - Consider working with a nutritionist
                - Start with low-impact exercises
                - Focus on sustainable lifestyle changes
                """)
        else:
            st.info("Please fill out the form and make a prediction to see the results.")

if __name__ == "__main__":
    main()
