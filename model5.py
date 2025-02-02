import streamlit as st
from transformers import pipeline
# Load T5 model for emotion classification
emotion_classifier = pipeline("text-classification", model="mrm8488/t5-base-finetuned-emotion")
def classify_query(text):
     """Classify query as Complaint or Normal Query using T5 model for emotion classification."""
     text = text.strip()
     # Get emotion prediction    
     result = emotion_classifier(text)[0]
     emotion = result["label"]
     confidence = result["score"]
    # Map emotion to Complaint or Normal Query  
     complaint_emotions = ["anger", "sadness", "fear", "disgust", "frustration"]
     classification = "Complaint" if emotion in complaint_emotions else "Normal Query"
     return classification, emotion, confidence
# Streamlit UI
st.set_page_config(page_title="Query Classifier", layout="centered", initial_sidebar_state='collapsed')
st.header("Query Classification with T5 (Emotion)")
# User Input
input_text = st.text_area("Enter your query:", height=200)
# Classify Button
if st.button("Classify Query"):
     if input_text.strip():
        classification, emotion, confidence = classify_query(input_text)
        st.subheader(f"**Classification:** {classification}")
        st.write(f"**Emotion:** {emotion.capitalize()} (Confidence: {round(confidence * 100, 2)}%)")
     else:
       st.warning("Please enter text to classify!")