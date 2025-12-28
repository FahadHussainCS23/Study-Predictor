import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Load the "Brain" you created in Step 3
model = joblib.load('personality_model.pkl')

# 2. Set up the Page Design
st.set_page_config(page_title="Study Predictor", page_icon="📚")

st.title("🧠 Personalized Study Predictor")
st.write("This tool uses Machine Learning to recommend study techniques based on your Big Five personality traits.")

st.divider()

# 3. Create the Input Section (Sliders)
st.sidebar.header("Input Personality Scores")
st.sidebar.write("Rate yourself from 0.0 (Low) to 1.0 (High)")

# These variables (o, c, e, a, n) must match the order in your training data
o = st.sidebar.slider("Openness to Experience", 0.0, 1.0, 0.5)
c = st.sidebar.slider("Conscientiousness", 0.0, 1.0, 0.5)
e = st.sidebar.slider("Extraversion", 0.0, 1.0, 0.5)
a = st.sidebar.slider("Agreeableness", 0.0, 1.0, 0.5)
n = st.sidebar.slider("Neuroticism", 0.0, 1.0, 0.5)

# 4. Prediction Logic
if st.button("Generate My Study Plan"):
    # Arrange inputs into the format the model expects
    features = np.array([[o, c, e, a, n]])
    prediction = model.predict(features)
    
    # List of names in the exact order of your CSV columns
    tech_names = [
        'Spaced Repetition', 'Active Recall', 'Interleaving', 
        'Elaborative Interrogation', 'Feynman Technique', 'Mind Mapping'
    ]
    
    # 5. Show Results
    st.subheader("Your Personalized Recommendations:")
    
    # Get the first (and only) row of predictions
    results = prediction[0]
    
    if sum(results) == 0:
        st.warning("Based on these scores, no specific technique is a strong match. Try adjusting your traits slightly.")
    else:
        for i in range(len(tech_names)):
            if results[i] == 1:
                st.success(f"**Recommended:** {tech_names[i]}")

st.divider()
st.caption("Developed by Arpita & Fahad | Data Science Project 2024")