import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

target_names = ["Setosa", "Versicolor", "Virginica"]

st.title("Iris Flower Classification App")
st.write("Enter measurements below to predict the Iris flower species.")

sl = st.number_input("Sepal Length", 0.0, 10.0, 5.1)
sw = st.number_input("Sepal Width", 0.0, 10.0, 3.5)
pl = st.number_input("Petal Length", 0.0, 10.0, 1.4)
pw = st.number_input("Petal Width", 0.0, 10.0, 0.2)

if st.button("Predict"):
    sample = np.array([[sl, sw, pl, pw]])
    sample_scaled = scaler.transform(sample)
    prediction = model.predict(sample_scaled)[0]
    
    st.success(f"Predicted Species: **{target_names[prediction]}**")
