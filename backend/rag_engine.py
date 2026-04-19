import os
import requests
from pathlib import Path
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    StorageContext, 
    Settings,
    Document
)
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
from llama_index.readers.file import PyMuPDFReader
import chromadb

load_dotenv()

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
PDF_URL = "https://www.nasa.gov/wp-content/uploads/2018/09/nasa_systems_engineering_handbook_0.pdf"
PDF_PATH = DATA_DIR / "nasa_systems_engineering_handbook.pdf"
CHROMA_DIR = Path("chroma_db")

# Initialize Models
groq_key = os.getenv("GROQ_API_KEY")

# LLM: Groq (Llama 3.3 70B) - Much faster and 100% stable
print("Initializing Groq LLM (Llama 3.3 70B)...")
Settings.llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_key)

# EMBEDDINGS: Local (Bypasses API 404s)
print("Loading local embedding model (BAAI/bge-small-en-v1.5)...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

Settings.chunk_size = 512
Settings.chunk_overlap = 50

def download_pdf():
    if not PDF_PATH.exists():
        print(f"Downloading NASA Handbook...")
        response = requests.get(PDF_URL)
        with open(PDF_PATH, "wb") as f:
            f.write(response.content)

def get_index():
    if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
        return None
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = db.get_or_create_collection("nasa_handbook")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return VectorStoreIndex.from_vector_store(vector_store)

def ingest_document():
    download_pdf()
    print("Parsing PDF...")
    loader = PyMuPDFReader()
    documents = loader.load(file_path=str(PDF_PATH))
    
    print("Creating nodes...")
    node_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 1024, 512])
    nodes = node_parser.get_nodes_from_documents(documents)
    leaf_nodes = get_leaf_nodes(nodes)
    
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))
    chroma_collection = db.get_or_create_collection("nasa_handbook")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    storage_context.docstore.add_documents(nodes)
    
    print("Indexing locally...")
    index = VectorStoreIndex(leaf_nodes, storage_context=storage_context, show_progress=True)
    return index

def get_query_engine():
    index = get_index()
    if not index:
        index = ingest_document()
        
    # Standard query engine with attribution-compliant persona
    return index.as_query_engine(
        similarity_top_k=5,
        system_prompt=(
            "You are a Technical Systems Engineering Assistant. Your responses are generated "
            "based on the public domain NASA Systems Engineering Handbook (SP-2016-6105 Rev2). "
            "Clearly attribute information to the source material (e.g., 'The handbook states...') "
            "and ALWAYS include Section and Page citations. You are an AI assistant, not an official NASA system."
        )
    )

if __name__ == "__main__":
    ingest_document()
