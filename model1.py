import streamlit as st
from transformers import pipeline
# Load sentiment analysis model
sentiment_classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
def classify_query(text):
    """Classify query as Complaint or Normal Query based on sentiment."""
    text = text.lower().strip()
    # Preprocess input text       
    # # Get sentiment prediction   
    result = sentiment_classifier(text)[0]
    sentiment = result["label"]
    confidence = result["score"]
    # Map sentiment to query type   
    if "1 star" in sentiment or "2 stars" in sentiment:
        classification = "Complaint"
    else:
        classification = "Normal Query"
    return classification, sentiment, confidence
# Streamlit UI
st.set_page_config(page_title="Query Classifier", layout="centered", initial_sidebar_state='collapsed')
st.header("Query Classification Based on Sentiment")
# User Input
input_text = st.text_area("Enter your query:", height=100)
# Classify Button
if st.button("Classify Query"):
    if input_text.strip():
        classification, sentiment, confidence = classify_query(input_text)
        st.subheader(f"**Classification:** {classification}")
        st.write(f"**Sentiment:** {sentiment} (Confidence: {round(confidence * 100, 2)}%)")
    else:
        st.warning("Please enter text to classify!")