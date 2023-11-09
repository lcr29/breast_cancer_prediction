import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('breast_cancer_prediction_model.h5')

st.set_page_config(layout="wide")

st.image('header_image.png', width=150)  
st.title('Breast Cancer Prediction')

st.markdown("""
    Enter the details of the medical diagnostic to predict the breast cancer diagnosis.
    Adjust the values of each feature using the input fields below.
""")

# Sidebar for feature input
st.sidebar.header('Input Features')
input_features = []

# List of feature names based on your provided information, excluding 'id' and 'diagnosis'
feature_names = ['radius_mean', 'texture_mean', 'smoothness_mean', 'compactness_mean',
                 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
                 'radius_se', 'texture_se', 'smoothness_se', 'compactness_se',
                 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                 'texture_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst',
                 'symmetry_worst', 'fractal_dimension_worst']

# Assuming all features are numerical and the user should enter a value
for feature_name in feature_names:
    value = st.sidebar.number_input(feature_name, min_value=0.0, step=0.001, value=0.0)
    input_features.append(value)

# Button to make the prediction
if st.sidebar.button('Predict Diagnosis'):
    input_array = np.array(input_features).reshape(1, -1)
    prediction = model.predict(input_array)
    if prediction[0] == 1:
        st.error('The diagnosis is malignant.')
        st.image('risk.png', caption='Immediate call to action: please see your medical professional for in-depth studies', width=300)
    else:
        st.success('The diagnosis is benign.')
        st.image('accept.png', caption='Everything is in order: go to your medical professional to confirm it', width=300)


# About section
expander = st.expander("About this app")
expander.write("""
    This app uses a machine learning model to predict the likelihood of breast cancer based on diagnostic details.
    It is important to note that this prediction is not a substitute for professional medical advice.

     **Model:**
    - The prediction model is based on Neural Networks, trained to detect patterns indicative malignant tumors.
""")