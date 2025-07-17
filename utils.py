import os
import fitz  # PyMuPDF
import chromadb
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import tempfile

# 🔐 Load environment variables (e.g., GROQ_API_KEY from .env file)
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"  # ✅ Groq endpoint (OpenAI compatible)

# 🧠 Initialize ChromaDB client with persistent storage for production
PERSISTENT_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
os.makedirs(PERSISTENT_DIR, exist_ok=True)

# Initialize global variables
model = None
chroma_client = None
collection = None

def initialize_services():
    global model, chroma_client, collection
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=PERSISTENT_DIR)
        try:
            collection = chroma_client.get_collection(name="research_chunks")
        except ValueError:
            collection = chroma_client.create_collection(name="research_chunks")
            
        # Initialize the embedding model
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return True
    except Exception as e:
        print(f"Error initializing services: {e}")
        return False

# Initialize services
initialize_services()

# 📄 Extract text from PDF using PyMuPDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# 📚 Break long text into overlapping chunks
def chunk_text(text, chunk_size=300, overlap=50):
    if not model or not collection:
        if not initialize_services():
            raise RuntimeError("Failed to initialize services")
            
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]
    
    try:
        # Add chunks to ChromaDB collection with unique IDs
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=model.encode(chunks).tolist()
        )
    except Exception as e:
        print(f"Warning: Failed to add chunks to ChromaDB: {e}")
    
    return chunks

# 🔎 Use ChromaDB to find top-k most similar chunks to the query
def search_top_chunks(query, chunks, embeddings=None, k=3):
    if not model or not collection:
        if not initialize_services():
            return chunks[:min(k, len(chunks))]  # Fallback to simple slicing
            
    try:
        results = collection.query(
            query_texts=[query],
            n_results=min(k, len(chunks))
        )
        return [chunks[int(doc_id.split('_')[1])] for doc_id in results['ids'][0]]
    except Exception as e:
        print(f"Search error: {e}")
        return chunks[:min(k, len(chunks))]  # Fallback to simple slicing

# 🤖 Call Groq's LLM to answer using the retrieved context
def ask_groq_llm(context, question):
    prompt = f"""You are a helpful healthcare research assistant. Use only the context below to answer the user's question.

Context:
{context}

Question:
{question}

Answer:"""

    response = openai.ChatCompletion.create(
        model="llama-3.3-70b-versatile",  # ✅ Production-supported Groq model
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()