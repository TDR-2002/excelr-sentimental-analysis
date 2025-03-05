import streamlit as st
import pickle
import numpy as np

def load_model():
    with open("model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("vectorizer.pkl", "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

def predict_sentiment(text, model, vectorizer):
    # Ensure text input is processed as a string
    if isinstance(text, str):  
        text = text.lower().strip()  # Apply lowercasing and strip spaces
        transformed_text = vectorizer.transform([text])  # Vectorize text
        prediction = model.predict(transformed_text)  # Predict sentiment
        return "Positive" if prediction == 1 else "Negative"
    else:
        return "Invalid input: Please enter text."


st.title("Sentiment Analysis App")
st.write("Enter text to analyze its sentiment.")

model, vectorizer = load_model()

user_input = st.text_area("Enter your text here:")
if st.button("Analyze Sentiment"):
    if user_input:
        result = predict_sentiment(user_input, model, vectorizer)
        st.write(f"Sentiment: {result}")
    else:
        st.write("Please enter some text.")
