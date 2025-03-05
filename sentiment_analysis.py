import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')

# Load and preprocess data function
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['text', 'title', 'rating']]
    df = df.drop_duplicates()
    return df

# Stemming function
def stemming(content):
    ps = PorterStemmer()
    stemmed_text = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_text = stemmed_text.lower()
    stemmed_text = stemmed_text.split()
    stemmed_text_filtered = []
    for i in stemmed_text:
        if i not in stopwords.words('english'):
            stemmed_text_filtered.append(ps.stem(i))
    stemmed_text = ' '.join(stemmed_text_filtered)
    return stemmed_text

# Train model function
def train_model(df):
    # Stemming title and text
    df['Stemmed_Text'] = df['text'].apply(stemming)
    df['Stemmed_title'] = df['title'].apply(stemming)
    df['content'] = df['Stemmed_title'] + df['Stemmed_Text']

    # Extract ratings
    df['Ratings'] = df['rating'].apply(lambda r: float(r.split()[0]))

    # Create sentiment labels
    def sentiment(rate):
        if rate > 3:
            return 'Positive'
        elif rate < 3:
            return 'Negative'
        else:
            return 'Neutral'

    df['Sentiments'] = df["Ratings"].apply(sentiment)
    df["Sentiments"] = LabelEncoder().fit_transform(df["Sentiments"])

    # Prepare data for training
    x = df["content"]
    y = df['Sentiments']

    # Vectorize text
    vectorizer = TfidfVectorizer()
    x_train = vectorizer.fit_transform(x)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y)

    return model, vectorizer

# Streamlit app
def main():
    st.title('Sentiment Analysis App')
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Train model
        model, vectorizer = train_model(df)
        
        # User input
        user_input = st.text_area("Enter your review")
        
        if st.button('Predict Sentiment'):
            if user_input:
                # Preprocess and predict
                processed_input = vectorizer.transform([user_input])
                prediction = model.predict(processed_input)
                
                # Map predictions to labels
                sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
                result = sentiment_map[prediction[0]]
                
                st.success(f'Predicted Sentiment: {result}')

if __name__ == '__main__':
    main()
