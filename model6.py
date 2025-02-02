import streamlit as st
from transformers import pipeline
# Load BART model for multi-class classification (MNLI)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
def classify_query(text):
    """Classify query as Complaint or Normal Query using BART (MNLI) model."""
    text = text.strip()
    # Possible labels for classification    
    candidate_labels = ["Complaint", "Normal Query"]
    # Perform zero-shot classification    
    result = classifier(text, candidate_labels) 
    # Extract the predicted label and its score    
    label = result['labels'][0]
    confidence = result['scores'][0]
    return label, confidence
# Streamlit UI
st.set_page_config(page_title="Query Classifier", layout="centered", initial_sidebar_state='collapsed')
st.header("Query Classification with BART (MNLI)")
# User Input
input_text = st.text_area("Enter your query:", height=200)
# Classify Button
if st.button("Classify Query"):
     if input_text.strip():
        label, confidence = classify_query(input_text)
        st.subheader(f"**Classification:** {label}")
        st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
     else:
       st.warning("Please enter text to classify!")