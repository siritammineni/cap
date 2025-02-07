import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from langchain.evaluation import load_evaluator  
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from sentence_transformers import SentenceTransformer, util


ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"
#TOKEN = "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"
TOKEN = "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"

client = ChatCompletionsClient(endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN))
CHUNKERS = {
    "RecursiveTextSplitter": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
    "SentenceSplitter": RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,  
        separators=["\n", ". ", "? ", "! "],  
        length_function=len  
    ),
    "ParagraphSplitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
}


EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bge-base-en": "BAAI/bge-base-en"  
}

VECTOR_DB_DIR = "vector_dbs"

queries=[
    "What security features does Xfinity offer?",
    "What is Xfinity NOW Internet?",
    "What is the Xfinity NOW WiFi Pass?",
    "What are the additional terms for Xfinity Internet and Voice services?",
    "How can I check my data usage for finity Internet?",
    "What happens if I exceed the data consumption threshold?"
]

ground_truth= {
    "What security features does Xfinity offer?" : "Xfinity includes Advanced Security to protect against phishers, hackers, and other threats, while 5G home internet may offer limited security features.",
    "What is Xfinity NOW Internet?" : "Xfinity NOW Internet is a prepaid home internet service with unlimited data, WiFi equipment, and all taxes and fees included.",
    "What is the Xfinity NOW WiFi Pass?" : "The Xfinity NOW WiFi Pass provides access to millions of hotspots with no commitments, available for just $10 for each 30-day pass.",
    "What are the additional terms for Xfinity Internet and Voice services?": "The additional terms apply to Xfinity Internet and Voice services and are incorporated into the finity Residential Services Agreement. By using the services, you accept these terms, and any conflict between the Agreement and the Adiitional Terms will be governed by the Additional terms for internet and voice",
    "How can I check my data usage for finity Internet?" : "You can check your current data usage by logging into your finity account at xfinity.com or using the Xfinity My Account mobile app.",
    "What happens if I exceed the data consumption threshold?" : "If you exceed the data consumption threshold, Xfinity may notify you, and they reserve the right to adjust your data plan or usage thresholds as needed."
}


# Function to load FAISS index and assign correct embedding function
def load_vector_store(chunker_name, emb_name):

    print("in load vector")
    file_path = os.path.join(VECTOR_DB_DIR, f"{chunker_name}_{emb_name}.pkl")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Vector DB not found for {chunker_name} + {emb_name}")

    with open(file_path, "rb") as f:
        vector_store = pickle.load(f)

    # ✅ Assign the correct embedding function based on emb_name
    if emb_name in EMBEDDING_MODELS:
        model_path = EMBEDDING_MODELS[emb_name]  # Get the embedding model path
        model = SentenceTransformer(model_path, device="cpu")  # Load model on CPU
        vector_store.embedding_function = lambda text: model.encode([text], convert_to_numpy=True)[0]  # Encode single query
    else:
        raise ValueError(f"Embedding model {emb_name} not found in defined models.")
    print("leaving vector store")
    return vector_store


def retrieve_context(query, vector_store, top_k=3):
    start_time = time.time()
    
    try:
        docs = vector_store.similarity_search(query, k=top_k)
    except TypeError:  
        # ✅ Handle missing index_to_docstore_id issue
        print("⚠️ FAISS index is missing index_to_docstore_id. Attempting fallback retrieval...")
        query_embedding = vector_store.embedding_function(query)  # Embed query
        D, I = vector_store.index.search(np.array([query_embedding]), k=top_k)  # Search FAISS index

        docs = []
        for idx in I[0]:  
            if idx == -1:
                continue  # Skip invalid indices
            retrieved_text = "Unknown Document"  # Default text
            if hasattr(vector_store, "docstore") and vector_store.docstore:
                retrieved_text = vector_store.docstore.get(str(idx), "Unknown Document")
            docs.append(retrieved_text)

    retrieval_time = time.time() - start_time
    print("leaving retrieve")
    return "\n\n".join(docs), docs, retrieval_time


# Function to query GPT-4o
def query_gpt4o(user_query, vector_store):
    print("in query gpt")
    context, docs, retrieval_time = retrieve_context(user_query, vector_store)
    prompt = f"Use the following retrieved context to answer the query:\nContext:\n{context}\nQuery: {user_query}"
    
    start_time = time.time()
    response = client.complete(
        messages=[SystemMessage(content="You are a helpful assistant."), UserMessage(content=prompt)],
        temperature=1.0, top_p=1.0, max_tokens=500, model=MODEL_NAME
    )
    response_time = time.time() - start_time

    return response.choices[0].message.content, retrieval_time, response_time

# Function to calculate Semantic Similarity Score
def calculate_similarity_score(response, ground_truth):
    print("in similarity search")
    if not ground_truth:
        return None  # Skip if no ground-truth provided
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    response_embedding = model.encode(response, convert_to_tensor=True)
    truth_embedding = model.encode(ground_truth, convert_to_tensor=True)
    
    similarity_score = util.pytorch_cos_sim(response_embedding, truth_embedding).item()
    print("leaving similarity")
    return round(similarity_score, 3)

# Function to evaluate all chunking + embedding combinations
def evaluate():
    print("in eval")
    results = []

    for chunker_name in CHUNKERS.keys():
        for emb_name in EMBEDDING_MODELS.keys():
            print(f"🔍 Evaluating: {chunker_name} + {emb_name}")
            
            # Load precomputed FAISS index
            try:
                vector_store = load_vector_store(chunker_name, emb_name)
            except FileNotFoundError:
                continue  # Skip if the vector store does not exist

            for query in queries:
                response, retrieval_time, response_time = query_gpt4o(query, vector_store)
                similarity_score = calculate_similarity_score(response, ground_truth.get(query))

                results.append({
                    "Chunking Strategy": chunker_name,
                    "Embedding Model": emb_name,
                    "Query": query,
                    "Retrieval Time (s)": round(retrieval_time, 3),
                    "Response Time (s)": round(response_time, 3),
                    "Semantic Similarity Score": similarity_score,
                    "Response": response
                })

    df = pd.DataFrame(results)
    
    # Save the evaluation results
    df.to_csv(r"C:\Users\338565\venv\chatbot\chatbot\rag_evaluation_results 1.csv", index=False)
    print("✅ Results saved as r'C:\Users\338565\venv\chatbot\chatbot\rag_evaluation_results 1.csv'")
    
    return df

# Run and visualize evaluation results
if __name__ == "__main__":
    
    results_df = evaluate()