import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load('model.joblib')

st.title("HDB Resale Price Prediction")

towns = ['Tampines', 'Bedok', 'Punggol']
flat_types = ['2 Room', '3 Room', '4 Room', '5 Room', 'Executive']
storey_ranges = ['01 to 03', '04 to 06', '07 to 09']


town_selected = st.selectbox("Select Town", towns)
flat_type_selected = st.selectbox("Select Flat Type", flat_types)
storey_range_selected = st.selectbox("Select Storey Range", storey_ranges)
floor_area_selected = st.slider("Select Floor Area (sqm)", min_value=30, max_value=200, value=70)

if st.button("Predict HDB Price"):

    input_data = { 
        'town': town_selected,
        'flat_type': flat_type_selected,
        'storey_range': storey_range_selected,
        'floor_area': floor_area_selected
    }

df_input = pd.DataFrame({'town': [town_selected],
                        'flat_type': [flat_type_selected],
                        'storey_range': [storey_range_selected],
                        'floor_area_sqm': [floor_area_selected]})

df_input = pd.get_dummies(df_input, columns=['town', 'flat_type', 'storey_range'])

df_input = df_input.reindex(columns = model.feature_names_in_, fill_value=0)

y_unseen_pred = model.predict(df_input)[0]
st.success(f"The predicted HDB resale price is: ${y_unseen_pred:,.2f}")

st.markdown(f''' <style> .stApp {{
    background-image: url("https://cdn.pixabay.com/photo/2018/08/04/11/30/draw-3583548_1280.png");
    background-size: cover;}}</style>''', unsafe_allow_html=True)
