# Obesity Level Prediction Web Application

## Overview
This project implements a machine learning-based web application for predicting obesity levels based on various lifestyle and health factors. The application uses XGBoost for classification and is deployed using Streamlit, providing an interactive and user-friendly interface for predictions.

## Dataset Information
The project uses the "Obesity Dataset" which contains data about individuals' lifestyle and health factors. The dataset includes both raw and synthetic data, making it comprehensive for obesity level prediction. The target variable (NObeyesdad) has multiple classes representing different obesity levels.

### Dataset Features
- **Demographic Information**:
  - Gender
  - Age
  - Height
  - Weight
  - Family history of obesity

- **Eating Habits**:
  - FAVC (Frequent high caloric food consumption)
  - FCVC (Vegetable consumption frequency)
  - NCP (Number of main meals)
  - CAEC (Food between meals)
  - CH2O (Daily water consumption)

- **Physical Activity**:
  - FAF (Physical activity frequency)
  - TUE (Technology usage)
  - SCC (Calorie monitoring)

- **Transportation**:
  - CALC (Transportation mode)
  - MTRANS (Transportation time)

## Data Analysis and Visualizations

### Distribution of Obesity Levels
![Obesity Level Distribution](images/obesity_distribution.png)
*Distribution of different obesity levels in the dataset*

### Correlation Analysis
![Feature Correlations](images/correlation_matrix.png)
*Correlation matrix showing relationships between different features*

### Feature Importance
![Feature Importance](images/feature_importance.png)
*XGBoost model feature importance analysis*

### BMI Distribution
![BMI Distribution](images/bmi_distribution.png)
*Distribution of BMI across different obesity levels*

### Age vs Weight Relationship
![Age vs Weight](images/age_weight.png)
*Relationship between age and weight across different obesity levels*

## Features
- Interactive web interface for inputting personal health and lifestyle data
- Real-time obesity level prediction using XGBoost model
- BMI calculation and visualization
- Feature importance analysis
- Personalized health recommendations
- Responsive and modern UI design

## Technical Stack
- **Frontend**: Streamlit
- **Machine Learning**: XGBoost
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Model Serialization**: Joblib, Pickle

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd obesity-prediction
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Input your information in the form:
   - Personal Information (Age, Gender, Height, Weight)
   - Eating Habits
   - Physical Activity
   - Transportation Details

4. Click "Predict Obesity Level" to get your results

## Model Details

The application uses an XGBoost classifier trained on various features including:
- Demographic information
- Eating habits
- Physical activity levels
- Transportation patterns
- Lifestyle choices

### Model Performance
The XGBoost model was chosen for its ability to handle both numerical and categorical features effectively. The model provides:
- High accuracy in obesity level prediction
- Feature importance analysis
- Robust performance across different obesity categories

## Results Interpretation

The application provides:
1. Predicted obesity level
2. BMI calculation and category
3. Visual BMI gauge
4. Feature importance analysis
5. Personalized health recommendations based on BMI category

## Project Structure
```
obesity-prediction/
├── app.py              # Main Streamlit application
├── requirements.txt    # Project dependencies
├── xgb_model.json     # Trained XGBoost model
├── label_encoder.joblib # Label encoder for categorical variables
├── Obesity_Prediction.ipynb # Jupyter notebook with analysis and model training
├── images/            # Directory containing visualization images
│   ├── obesity_distribution.png
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── bmi_distribution.png
│   └── age_weight.png
└── README.md          # Project documentation
```

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- Dataset source: Obesity Dataset (Raw and Synthetic)
- XGBoost documentation
- Streamlit documentation 