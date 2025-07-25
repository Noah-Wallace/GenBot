# utils.py
import os
import fitz  # PyMuPDF
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import tempfile
from groq import Groq
from docx import Document
import io

# Initialize global variables
model = None
chroma_client = None
collection = None

def initialize_services():
    """Initialize ChromaDB and embedding model with proper error handling"""
    global model, chroma_client, collection
    
    # First, verify HF token is available
    hf_token = os.environ.get("HF_API_TOKEN")
    if not hf_token:
        print("⚠️ HF_API_TOKEN not found in environment variables")
        return False
        
    try:
        # Initialize the embedding model first
        print("Initializing embedding model...")
        model = SentenceTransformer(
            'sentence-transformers/all-MiniLM-L6-v2',
            token=hf_token  # Pass the token for authentication
        )
        print("✅ Embedding model initialized successfully")
        
        # Now initialize ChromaDB
        print("Initializing ChromaDB...")
        temp_dir = tempfile.mkdtemp()
        chroma_client = chromadb.PersistentClient(path=temp_dir)
        
        try:
            collection = chroma_client.get_collection(name="research_chunks")
            print("✅ Found existing ChromaDB collection")
        except:
            # Collection doesn't exist or error occurred, create a new one
            print("Creating new ChromaDB collection...")
            collection = chroma_client.create_collection(
                name="research_chunks",
                metadata={"description": "Document chunks for RAG"}
            )
            print("✅ Created new ChromaDB collection")
        
        print("✅ All services initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing services: {e}")
        # Continue without ChromaDB if initialization fails
        try:
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print("Initialized embedding model without ChromaDB")
            return True
        except Exception as e2:
            print(f"Failed to initialize embedding model: {e2}")
            return False

def extract_text(file_obj, file_path):
    """Extract text from various file formats"""
    try:
        extension = file_path.split('.')[-1].lower()
        
        if extension == 'pdf':
            return extract_pdf_text(file_obj)
        elif extension in ['docx', 'doc']:
            return extract_docx_text(file_obj)
        elif extension == 'txt':
            # Handle both file objects and bytes
            if hasattr(file_obj, 'read'):
                content = file_obj.read()
            else:
                content = file_obj
            
            if isinstance(content, bytes):
                return content.decode('utf-8', errors='ignore')
            return str(content)
        else:
            raise ValueError(f"Unsupported file type: {extension}")
            
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def extract_pdf_text(file_obj):
    """Extract text from PDF using PyMuPDF"""
    try:
        # Handle both file objects and bytes
        if hasattr(file_obj, 'read'):
            pdf_bytes = file_obj.read()
        else:
            pdf_bytes = file_obj
            
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_docx_text(file_obj):
    """Extract text from DOCX file"""
    try:
        # Handle file object
        if hasattr(file_obj, 'read'):
            file_content = file_obj.read()
        else:
            file_content = file_obj
        
        # Create a BytesIO object from the content
        docx_stream = io.BytesIO(file_content)
        doc = Document(docx_stream)
        
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        
        return text
    except Exception as e:
        print(f"Error extracting DOCX text: {e}")
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    """Break long text into overlapping chunks"""
    if not text or not text.strip():
        return []
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
    
    return chunks

def embed_chunks(chunks):
    """Create embeddings for text chunks"""
    if not chunks:
        return None
        
    # Initialize services if not already done
    if not model:
        if not initialize_services():
            print("Failed to initialize services for embedding")
            return None
    
    try:
        # Generate embeddings for all chunks
        embeddings = model.encode(chunks)
        
        # Store chunks and embeddings in ChromaDB if available
        if collection:
            try:
                # Clear existing data first
                try:
                    collection.delete(where={})
                except Exception:
                    pass  # Ignore if collection is empty
                
                # Add chunks to ChromaDB collection with unique IDs
                collection.add(
                    documents=chunks,
                    ids=[f"chunk_{i}" for i in range(len(chunks))],
                    embeddings=embeddings.tolist()
                )
                print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
            except Exception as e:
                print(f"Warning: Failed to add chunks to ChromaDB: {e}")
        
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def search_top_chunks(query, chunks, embeddings, k=5):
    """Find most relevant chunks using embeddings similarity"""
    if not chunks:
        return []
        
    # Initialize services if not already done
    if not model:
        if not initialize_services():
            return chunks[:min(k, len(chunks))]  # Fallback to simple slicing
    
    try:
        # Get query embedding
        query_embedding = model.encode([query])[0]
        
        # Calculate cosine similarity between query and all chunks
        similarities = []
        for chunk_embedding in embeddings:
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append(similarity)
        
        # Get indices of top k most similar chunks
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Filter by minimum similarity threshold
        threshold = 0.1
        top_chunks = []
        for idx in top_indices:
            if similarities[idx] >= threshold:
                top_chunks.append(chunks[idx])
        
        return top_chunks if top_chunks else chunks[:min(k, len(chunks))]
        
    except Exception as e:
        print(f"Search error: {e}")
        return chunks[:min(k, len(chunks))]  # Fallback to simple slicing

def ask_groq_llm(context, question):
    """Call Groq's LLM to answer using the retrieved context"""
    try:
        # Get API key from environment
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            return "Error: GROQ_API_KEY not found. Please set it in your Hugging Face Spaces settings."
        
        # Initialize Groq client
        client = Groq(api_key=groq_api_key)
        
        prompt = f"""You are a helpful research assistant. Use only the context below to answer the user's question accurately and comprehensively.

If the answer cannot be found in the provided context, please say so clearly.

Context:
{context}

Question:
{question}

Answer:"""

        # Make the API call
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama-3.1-70b-versatile",  # Updated to a current supported model
            max_tokens=1024,
            temperature=0.1
        )
        
        return chat_completion.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error calling Groq LLM: {e}")
        return f"Error: Unable to generate response. {str(e)}"

# Initialize services when module is imported
print("Initializing services...")
success = initialize_services()
if success:
    print("✅ Services initialized successfully")
else:
    print("⚠️ Services initialization failed - some features may not work")