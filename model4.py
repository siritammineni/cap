#limited query only
import streamlit as st
from transformers import pipeline
# Load BERTweet sentiment model
sentiment_classifier = pipeline("sentiment-analysis", model="finiteautomata/bertweet-base-sentiment-analysis")
def classify_query(text):
    """Classify query as Complaint or Normal Query using BERTweet sentiment model."""
    text = text.lower().strip()
    # Preprocess input text
    # Get sentiment prediction   
    result = sentiment_classifier(text)[0]
    sentiment = result["label"]
    confidence = result["score"]
    # Map sentiment to Complaint or Normal Query   
    classification = "Complaint" if sentiment == "NEG" else "Normal Query"  
    return classification, sentiment, confidence
# Streamlit UI
st.set_page_config(page_title="Query Classifier", layout="centered", initial_sidebar_state='collapsed')
st.header("Query Classification Based on Sentiment (BERTweet)")
# User Input
input_text = st.text_area("Enter your query:", height=100)
# Classify Button
if st.button("Classify Query"):
    if input_text.strip():
        classification, sentiment, confidence = classify_query(input_text)
        st.subheader(f"**Classification:** {classification}")
        st.write(f"**Sentiment:** {sentiment.capitalize()} (Confidence: {round(confidence * 100, 2)}%)")
    else:
        st.warning("Please enter text to classify!")