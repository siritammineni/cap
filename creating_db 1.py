import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

chunking_strategies = {
    "RecursiveTextSplitter": RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50),
    "SentenceSplitter": RecursiveCharacterTextSplitter(
        chunk_size=300, 
        chunk_overlap=50,  
        separators=["\n", ". ", "? ", "! "],  
        length_function=len  
    ),
    "ParagraphSplitter": RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
}

# Define embedding models
embedding_models = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "bge-base-en": "BAAI/bge-base-en" 
}

vector_dbs = "vector_dbs"

if not os.path.exists(vector_dbs):
    os.makedirs(vector_dbs)

def load_text_files(directory="scraped pages final"):
    print("🔹 Loading text files...")
    texts = []
    for file in os.listdir(directory):
        if file.endswith(".txt"):
            with open(os.path.join(directory, file), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)

def create_and_save_vector_store(text, chunker_name, chunker, emb_name, emb_model):
    print(f"Creating Vector Store for {chunker_name} + {emb_name}")

    # Chunk the text
    chunks = chunker.split_text(text)

    # Use Local Sentence Transformer for Embeddings
    model = SentenceTransformer(emb_model, device="cpu")  # No API needed
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    # ✅ Convert embeddings to a NumPy float32 array (FAISS Requirement)
    embeddings_array = np.array(embeddings).astype("float32")

    # Convert metadata into a list of dictionaries
    metadatas = [{"source": f"{chunker_name}_{i}"} for i in range(len(chunks))]

    # Use FAISS IndexFlatL2 for vector storage
    index = faiss.IndexFlatL2(embeddings_array.shape[1])  
    index.add(embeddings_array)  

    # Store embeddings & metadata using FAISS in LangChain
    vector_store = FAISS(index=index, embedding_function=model, docstore=None, index_to_docstore_id=None)

    # Save FAISS index
    save_path = os.path.join("vector_dbs", f"{chunker_name}_{emb_name}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(vector_store, f)
    
    print(f" Saved: {save_path}")

# Load and chunk text
raw_text = load_text_files()

# Generate and save vector stores
for chunker_name, chunker in chunking_strategies.items():
    for emb_name, emb_model in embedding_models.items():
            create_and_save_vector_store(raw_text, chunker_name, chunker, emb_name, emb_model)


