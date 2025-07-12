import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import openai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# 🔐 Load environment variables (e.g., GROQ_API_KEY from .env file)
load_dotenv()
openai.api_key = os.getenv("GROQ_API_KEY")
openai.api_base = "https://api.groq.com/openai/v1"  # ✅ Groq endpoint (OpenAI compatible)

# 🧠 Load embedding model once (local from Hugging Face)
model = SentenceTransformer('all-MiniLM-L6-v2')

# 📄 Extract text from PDF using PyMuPDF
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# 📚 Break long text into overlapping chunks
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size - overlap)]

# 🔗 Convert text chunks into embeddings
def embed_chunks(chunks):
    return model.encode(chunks)

# 🔍 Convert question to embedding
def embed_query(query):
    return model.encode([query])[0]

# 🔎 Use FAISS to find top-k most similar chunks to the query
def search_top_chunks(query, chunks, embeddings, k=3):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    query_vec = np.array([embed_query(query)]).astype("float32")
    _, I = index.search(query_vec, k)
    return [chunks[i] for i in I[0]]

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