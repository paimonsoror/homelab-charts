import requests
import logging
import http.client as http_client
import httpx

import os
os.environ["LANGCHAIN_TELEMETRY_ENABLED"] = "false"

from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader

from chromadb import Client
from langchain_ollama import OllamaEmbeddings
from chromadb.config import Settings

######
# Enable HTTPConnection debug logging
http_client.HTTPConnection.debuglevel = 1

# Configure root logger to output DEBUG level logs
logging.basicConfig(level=logging.DEBUG)

# Enable logging for urllib3 (used by requests)
logging.getLogger("urllib3").setLevel(logging.DEBUG)
logging.getLogger("urllib3").propagate = True
######

class ChromaV1Wrapper:
    def __init__(self, api_host: str, api_port: int = 443, ssl: bool = True, collection_name: str = "default"):
        settings = Settings(
	        allow_reset=True,
            chroma_api_impl="chromadb.api.fastapi.FastAPI",
            chroma_server_host=api_host,
            chroma_server_http_port=api_port,
            chroma_server_ssl_enabled=ssl,
            chroma_server_ssl_verify=False,
            anonymized_telemetry=False
        )
        self.client = Client(settings=settings)
        self.collection = self.client.get_or_create_collection(name=collection_name)

        embedding = OllamaEmbeddings(
            model="mistral",
            base_url="http://truenas.homelab:30068",
            client_kwargs= {
                "verify":False
            }
        )

        self.embeddings_model = embedding
    def add_documents(self, docs: list[str], ids: list[str] = None):
        # Generate embeddings
        embeddings = self.embeddings_model.embed_documents(docs)
        # Insert into collection
        self.collection.add(documents=docs, embeddings=embeddings, ids=ids)
        print("[Ollama SSL] Patched Embeddings Done!!")
    def similarity_search(self, query: str, k: int = 3):
        # Embed query
        query_embedding = self.embeddings_model.embed_query(query)
        # Query collection
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        # results is a dict with keys like 'documents', 'ids', etc.
        return results.get('documents', [])

# Load documents
loader = TextLoader("cluster.yaml")
docs = loader.load()
texts = [doc.page_content for doc in docs]

# Initialize your wrapper
chroma_wrapper = ChromaV1Wrapper(
    api_host="chromadb.homelab",
    api_port=443,
    ssl=True,
    collection_name="k8s-docs"
)

# Add documents (with optional unique IDs)
chroma_wrapper.add_documents(texts, ids=[f"doc{i}" for i in range(len(texts))])

# Query example
query = "Are there any failed pods?"
results = chroma_wrapper.similarity_search(query, k=3)

# Print results
print("Retrieved docs:")
for doc in results:
    print(doc)

# Optionally, use Ollama LLM to answer based on retrieved docs
llm = Ollama(
  model="mistral", 
  base_url="http://truenas.homelab:30068"
)
response = llm.invoke(f"Answer the question based on:\n\n{results}\n\nQuestion: {query}")
print("LLM response:", response)

