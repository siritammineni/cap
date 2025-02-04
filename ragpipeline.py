import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


#llm creds
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o"
TOKEN = "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"

#embedding creds
AZURE_OPENAI_API_KEY = "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"
AZURE_OPENAI_ENDPOINT = "https://models.inference.ai.azure.com"
AZURE_EMBEDDING_MODEL = "text-embedding-3-small"

client = ChatCompletionsClient(
   endpoint=ENDPOINT, credential=AzureKeyCredential(TOKEN)
)

FAISS_INDEX_PATH = "faiss index"  

def load_vector_store():
   
   if not os.path.exists(FAISS_INDEX_PATH):
       raise ValueError("FAISS index not found! Run `setup_knowledge_base.py` first.")
   print("✅ Loading KB from FAISS index...")

   return FAISS.load_local(
            FAISS_INDEX_PATH,
            OpenAIEmbeddings(
                model=AZURE_EMBEDDING_MODEL,
                openai_api_key=AZURE_OPENAI_API_KEY,
                openai_api_base=AZURE_OPENAI_ENDPOINT
            ),
            allow_dangerous_deserialization=True
        )



def retrieve_context(query, vector_store, top_k=3):
   
   docs = vector_store.similarity_search(query, k=top_k)
   results = []
   for doc in docs:
       source = doc.metadata["source"]  
       results.append(f" **Source:** {source}\n **Content:** {doc.page_content}")
   return "\n\n".join(results), docs


def query_gpt4o(user_query, vector_store):
   
   context, docs = retrieve_context(user_query, vector_store)
   sources = "\n".join(set(doc.metadata["source"] for doc in docs))  
   prompt = f"""Use the following retrieved context to answer the query:
   Context:
   {context}
   Query: {user_query}
   """
   response = client.complete(
       messages=[
           SystemMessage(content="You are a helpful assistant."),
           UserMessage(content=prompt)
       ],
       temperature=1.0,
       top_p=1.0,
       max_tokens=1000,
       model=MODEL_NAME
   )
   return response.choices[0].message.content, sources


if __name__ == "__main__":
   vector_store = load_vector_store()  
   print("Chatbot is ready! Ask your questions.")
   while True:
       query = input("\nAsk a question (or type 'exit' to quit): ")
       if query.lower() == "exit":
           break
       response, sources = query_gpt4o(query, vector_store)
       print("\nResponse:\n", response)
       print("\n Source:  \n", sources)