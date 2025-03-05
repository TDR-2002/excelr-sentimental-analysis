import streamlit as st
import pickle
import pandas as pd
import joblib


# Load the trained model
with open("new.pkl", "rb") as file:
    model = joblib.load(file)

# Streamlit UI
st.title("Exceltr Sentiment Analysis")

uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx", "csv"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)  # Adjust based on your file format
    st.write("Uploaded Data:")
    st.write(df.head())

    # Assuming you have a column 'text' for sentiment analysis
    if "text" in df.columns:
        df["prediction"] = model.predict(df["text"])  # Modify as per your model input
        st.write("Predictions:")
        st.write(df)
    else:
        st.write("No 'text' column found in the file.")
