import streamlit as st
from transformers import pipeline
# Load FinBERT sentiment model
sentiment_classifier = pipeline("sentiment-analysis", model="ProsusAI/finbert")
def classify_query(text):
    """Classify query as Complaint or Normal Query using FinBERT sentiment model."""
    text = text.lower().strip()
    # Preprocess input text
    # Get sentiment prediction    
    result = sentiment_classifier(text)[0]
    sentiment = result["label"]
    confidence = result["score"]
    # Map sentiment to Complaint or Normal Query    
    classification = "Complaint" if sentiment == "negative" else "Normal Query"
    return classification, sentiment, confidence
# Streamlit UI
st.set_page_config(page_title="Query Classifier", layout="centered", initial_sidebar_state='collapsed')
st.header("Query Classification Based on Sentiment (FinBERT)")
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