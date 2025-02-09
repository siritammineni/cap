import easyocr  # OCR library for text extraction
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import re
from langgraph.graph import StateGraph
from typing import Dict
 
# Load the CLIP model and processor
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)
 
# Initialize EasyOCR reader
ocr_reader = easyocr.Reader(['en'])  # Initialize OCR reader for English text
 
AgentState = Dict[str, str]
 
def extract_text_from_bill(image_path: str) -> str:
    """Extracts text from the image using EasyOCR."""
    result = ocr_reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in result])  # Concatenate all text detected
    return extracted_text.strip()
 
def analyze_image_content(image_path: str) -> str:
    """Uses CLIP to understand the content of an image and match it to predefined Xfinity-related subjects."""
    # Open image and preprocess it for CLIP
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True)
    
    # Get CLIP model outputs for the image
    outputs = model.get_image_features(**inputs)
    
    # CLIP provides an image embedding, we can compare this with some predefined text descriptions if needed
    image_embedding = outputs  # This is the image representation
    
    # Define text descriptions related to Xfinity topics that the chatbot should recognize
    text_descriptions = [
        "account details", "bill payment issues", "service outage", "customer support", 
        "internet issues", "hardware setup", "WiFi troubleshooting", "cable TV support", 
        "remote control issues", "service upgrade", "promotional offers", 
        "A dog playing in the park", "A cat sitting on the sofa", "A beautiful sunset over the mountains", 
        "A bowl of fresh vegetables",    "A person riding a bicycle", "An office workspace with a laptop", 
        "A car driving on the highway", "A historic monument", "A group of people having a meeting", 
        "A child playing with toys"
    ]
    
    text_inputs = processor(text=text_descriptions, padding=True, return_tensors="pt")
    text_embeddings = model.get_text_features(**text_inputs)
    
    # Cosine similarity to find the best match
    similarity = torch.cosine_similarity(image_embedding, text_embeddings, dim=-1)
    best_match_index = similarity.argmax().item()
    
    return text_descriptions[best_match_index]  # Return the best matching subject
 
def analyze_bill(state: AgentState) -> AgentState:
    image_path = "sample_image.jpg"  # Replace with your actual file path
    extracted_text = extract_text_from_bill(image_path)
    
    # Analyze the content of the image using CLIP for subject matching
    image_content_description = analyze_image_content(image_path)
 
    # Regular expressions to extract key bill details
    date_pattern = r"\b\d{1,2}/\d{1,2}/\d{4}\b"  # Matches dates like 12/01/2025
    amount_pattern = r"Total\s+\$?(\d+\.\d{2})"   # Matches 'Total $99.99'
    invoice_pattern = r"Invoice\s+#?(\d+)"        # Matches 'Invoice #12345'
 
    date_match = re.search(date_pattern, extracted_text)
    amount_match = re.search(amount_pattern, extracted_text)
    invoice_match = re.search(invoice_pattern, extracted_text)
 
    # Store extracted details in state
    state["invoice_number"] = invoice_match.group(1) if invoice_match else "Not Found"
    state["date"] = date_match.group() if date_match else "Not Found"
    state["total_amount"] = amount_match.group(1) if amount_match else "Not Found"
    state["full_text"] = extracted_text
    state["image_content"] = image_content_description  # Store the image content description
 
    return state
 
# Create LangGraph workflow
graph = StateGraph(AgentState)
graph.add_node("bill_analysis", analyze_bill)
graph.set_entry_point("bill_analysis")
graph.set_finish_point("bill_analysis")
bill_analysis_agent = graph.compile()
 
def main():
    print("Starting the Bill Analysis Agent")
    result = bill_analysis_agent.invoke({})
    
    print("\nExtracted Bill Details:")
    print(f"Invoice Number: {result.get('invoice_number', 'N/A')}")
    print(f"Date: {result.get('date', 'N/A')}")
    print(f"Total Amount: {result.get('total_amount', 'N/A')}")
    print("\nFull Extracted Text:\n", result.get("full_text", ""))
    print(f"\nImage Content Understanding: {result.get('image_content', 'N/A')}")
 
if __name__ == "__main__":
    main()
 