# app.py

from flask import Flask, render_template, request, flash, redirect, url_for
import numpy as np
from werkzeug.utils import secure_filename
import os
from utils import extract_text, chunk_text, embed_chunks, search_top_chunks, ask_groq_llm

app = Flask(__name__)
#app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB
app.config['UPLOAD_FOLDER'] = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}

# Ensure upload directory exists
#os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
upload_folder = 'uploads'
if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_files(files):
    """Validate uploaded files"""
    if not files or all(f.filename == '' for f in files):
        return False, "No files selected"
    
    for file in files:
        if file.filename == '':
            continue
        if not allowed_file(file.filename):
            return False, f"File type not allowed: {file.filename}"
    
    return True, "Files are valid"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get question first
            question = request.form.get('question', '').strip()
            if not question:
                flash('Please enter a question', 'error')
                return redirect(url_for('index'))
            
            # Get uploaded files
            files = request.files.getlist('pdf')  # Consider renaming to 'files' for clarity
            
            # Validate files
            is_valid, message = validate_files(files)
            if not is_valid:
                flash(message, 'error')
                return redirect(url_for('index'))
            
            # Process files
            all_text = ""
            processed_files = []
            
            for file in files:
                if file.filename == '':
                    continue
                    
                try:
                    # Secure the filename
                    filename = secure_filename(file.filename)
                    
                    # Extract text from file
                    text = extract_text(file)
                    
                    if text.strip():  # Only add non-empty text
                        all_text += f"\n\n--- Content from {filename} ---\n\n"
                        all_text += text
                        processed_files.append(filename)
                    else:
                        flash(f'No text could be extracted from {filename}', 'warning')
                        
                except Exception as e:
                    flash(f'Error processing {file.filename}: {str(e)}', 'error')
                    continue
            
            # Check if we have any text to process
            if not all_text.strip():
                flash('No text could be extracted from any of the uploaded files', 'error')
                return redirect(url_for('index'))
            
            # Process the combined text
            try:
                chunks = chunk_text(all_text)
                if not chunks:
                    flash('No text chunks could be created from the files', 'error')
                    return redirect(url_for('index'))
                
                embeddings = embed_chunks(chunks)
                if embeddings is None or len(embeddings) == 0:
                    flash('Failed to create embeddings from text chunks', 'error')
                    return redirect(url_for('index'))
                
                # Search for relevant chunks
                top_chunks = search_top_chunks(
                    question, 
                    chunks, 
                    np.array(embeddings).astype("float32")
                )
                
                if not top_chunks:
                    flash('No relevant information found in the uploaded files', 'warning')
                    return redirect(url_for('index'))
                
                # Prepare context for LLM
                context = "\n\n".join(top_chunks)
                
                # Get answer from LLM
                answer = ask_groq_llm(context, question)
                
                if not answer:
                    flash('Failed to generate answer from the LLM', 'error')
                    return redirect(url_for('index'))
                
                return render_template(
                    "result.html", 
                    question=question, 
                    answer=answer, 
                    top_chunks=top_chunks,
                    processed_files=processed_files,
                    total_files=len(processed_files)
                )
                
            except Exception as e:
                flash(f'Error processing text: {str(e)}', 'error')
                return redirect(url_for('index'))
                
        except Exception as e:
            flash(f'An unexpected error occurred: {str(e)}', 'error')
            return redirect(url_for('index'))
    
    return render_template("index.html")

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    flash('File is too large. Maximum size is 50MB.', 'error')
    return redirect(url_for('index'))

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle unexpected errors"""
    flash(f'An error occurred: {str(e)}', 'error')
    return redirect(url_for('index'))

@app.route('/test-result')
def test_result():
    print("✅ Rendering result.html")
    print("Question:", question)
    print("Answer:", answer)
    print("Chunks:", top_chunks)
    print("Files:", processed_files)

    return render_template("result.html", question="What is AI?", answer="AI is artificial intelligence.", top_chunks=["Chunk 1", "Chunk 2"])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))