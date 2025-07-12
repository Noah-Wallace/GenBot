# pdf_chunker.py

import fitz  # PyMuPDF
from openai import OpenAI
import faiss
import numpy as np
import tiktoken

openai_api_key = "your-openai-api-key"

def extract_text_from_pdf(path):
    doc = fitz.open(path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def get_embedding(text):
    # Replace with openai.Embedding.create() if needed
    response = OpenAI(api_key=openai_api_key).embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response['data'][0]['embedding']

def build_faiss_index(chunks):
    dimension = 1536  # For ada-002
    index = faiss.IndexFlatL2(dimension)
    embeddings = [get_embedding(c) for c in chunks]
    index.add(np.array(embeddings).astype('float32'))
    return index, embeddings

def save_chunks(path):
    text = extract_text_from_pdf(path)
    chunks = chunk_text(text)
    return chunks
