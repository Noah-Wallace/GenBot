# app.py - Gradio Version of RAG Document Q&A

import gradio as gr
import numpy as np
import os
from typing import List, Tuple, Optional
from utils import extract_text, chunk_text, embed_chunks, search_top_chunks, ask_groq_llm

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}

def validate_file(file_path: str) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if not file_path:
        return False, "No file provided"
    
    # Check file size
    if os.path.getsize(file_path) > MAX_FILE_SIZE:
        return False, "File is too large (max 50MB)"
    
    # Check file extension
    extension = file_path.split('.')[-1].lower()
    if extension not in ALLOWED_EXTENSIONS:
        return False, f"File type '.{extension}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    
    return True, "File is valid"

def process_documents_and_answer(files: List[str], question: str) -> Tuple[str, str, str]:
    """
    Process uploaded documents and answer the question
    
    Args:
        files: List of file paths
        question: User's question
        
    Returns:
        Tuple of (answer, sources_info, error_message)
    """
    try:
        # Validate inputs
        if not question or not question.strip():
            return "", "", "❌ Please enter a question"
        
        if not files:
            return "", "", "❌ Please upload at least one document"
        
        # Process files
        all_text = ""
        processed_files = []
        errors = []
        
        for file_path in files:
            try:
                # Validate file
                is_valid, message = validate_file(file_path)
                if not is_valid:
                    errors.append(f"❌ {os.path.basename(file_path)}: {message}")
                    continue
                
                # Extract text from file
                with open(file_path, 'rb') as f:
                    # Create a file-like object that extract_text expects
                    text = extract_text(f, file_path)
                
                if text and text.strip():
                    filename = os.path.basename(file_path)
                    all_text += f"\n\n--- Content from {filename} ---\n\n"
                    all_text += text
                    processed_files.append(filename)
                else:
                    errors.append(f"⚠️ {os.path.basename(file_path)}: No text could be extracted")
                    
            except Exception as e:
                errors.append(f"❌ {os.path.basename(file_path)}: {str(e)}")
                continue
        
        # Check if we have any text to process
        if not all_text.strip():
            error_msg = "❌ No text could be extracted from any files"
            if errors:
                error_msg += "\n\nErrors:\n" + "\n".join(errors)
            return "", "", error_msg
        
        # Process the combined text
        try:
            # Create chunks
            chunks = chunk_text(all_text)
            if not chunks:
                return "", "", "❌ No text chunks could be created from the files"
            
            # Create embeddings
            embeddings = embed_chunks(chunks)
            if embeddings is None or len(embeddings) == 0:
                return "", "", "❌ Failed to create embeddings from text chunks"
            
            # Search for relevant chunks
            top_chunks = search_top_chunks(
                question, 
                chunks, 
                np.array(embeddings).astype("float32")
            )
            
            if not top_chunks:
                return "", "", "❌ No relevant information found in the uploaded files"
            
            # Prepare context for LLM
            context = "\n\n".join(top_chunks)
            
            # Get answer from LLM
            answer = ask_groq_llm(context, question)
            
            if not answer:
                return "", "", "❌ Failed to generate answer from the LLM"
            
            # Prepare sources information
            sources_info = f"""
**📁 Sources Used:**
- **Files processed:** {len(processed_files)}
- **Files:** {', '.join(processed_files)}
- **Text chunks found:** {len(top_chunks)}

**📝 Relevant excerpts:**
"""
            
            for i, chunk in enumerate(top_chunks[:3], 1):  # Show top 3 chunks
                preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
                sources_info += f"\n**Chunk {i}:**\n{preview}\n"
            
            # Add any errors as warnings
            error_msg = ""
            if errors:
                error_msg = "⚠️ Some files had issues:\n" + "\n".join(errors)
            
            return answer, sources_info, error_msg
            
        except Exception as e:
            return "", "", f"❌ Error processing text: {str(e)}"
            
    except Exception as e:
        return "", "", f"❌ An unexpected error occurred: {str(e)}"

def create_interface():
    """Create and return the Gradio interface"""
    
    with gr.Blocks(
        title="📚 Document Q&A with RAG",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📚 Document Q&A with RAG
        
        Upload your documents (PDF, TXT, DOCX, DOC) and ask questions about their content. 
        The system uses Retrieval-Augmented Generation (RAG) to find relevant information and provide accurate answers.
        
        **Supported formats:** PDF, TXT, DOCX, DOC (max 50MB per file)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload
                files_input = gr.File(
                    label="📎 Upload Documents",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx", ".doc"],
                    height=150
                )
                
                # Question input
                question_input = gr.Textbox(
                    label="❓ Your Question",
                    placeholder="What would you like to know about the uploaded documents?",
                    lines=3,
                    max_lines=5
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "🔍 Ask Question",
                    variant="primary",
                    size="lg"
                )
                
            with gr.Column(scale=2):
                # Answer output
                answer_output = gr.Markdown(
                    label="💡 Answer",
                    value="Upload documents and ask a question to get started!",
                    height=300
                )
                
        # Sources and errors in expandable sections
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(
                    label="📚 Sources & Context",
                    visible=False
                )
                
                error_output = gr.Markdown(
                    label="⚠️ Warnings & Errors",
                    visible=False
                )
        
        # Handle form submission
        def process_and_display(files, question):
            if not files or not question:
                return (
                    "Please upload documents and enter a question.",
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            # Get file paths
            file_paths = [f.name for f in files] if files else []
            
            # Process documents and get answer
            answer, sources, errors = process_documents_and_answer(file_paths, question)
            
            # Prepare outputs
            if answer:
                formatted_answer = f"## 💡 Answer\n\n{answer}"
                sources_visible = True
                errors_visible = bool(errors)
            else:
                formatted_answer = errors if errors else "No answer generated."
                sources_visible = False
                errors_visible = False
            
            return (
                formatted_answer,
                gr.update(value=sources, visible=sources_visible),
                gr.update(value=errors, visible=errors_visible)
            )
        
        # Connect the submit button
        submit_btn.click(
            fn=process_and_display,
            inputs=[files_input, question_input],
            outputs=[answer_output, sources_output, error_output]
        )
        
        # Also allow Enter key in question box
        question_input.submit(
            fn=process_and_display,
            inputs=[files_input, question_input],
            outputs=[answer_output, sources_output, error_output]
        )
        
        # Example section
        gr.Markdown("""
        ## 📖 Example Questions
        
        Try asking questions like:
        - "What are the main points discussed in the document?"
        - "Can you summarize the key findings?"
        - "What does the document say about [specific topic]?"
        - "What are the recommendations mentioned?"
        """)
    
    return demo

# Create the interface
demo = create_interface()

if __name__ == "__main__":
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True
    )