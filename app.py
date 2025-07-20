# app.py - RAG Document Q&A Chatbot

import gradio as gr
import numpy as np
import os
from typing import List, Tuple, Optional
from utils import extract_text, chunk_text, embed_chunks, search_top_chunks, ask_groq_llm

# Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'doc'}

class DocumentProcessor:
    """Handles document processing and validation"""
    
    @staticmethod
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

    @staticmethod
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
                    is_valid, message = DocumentProcessor.validate_file(file_path)
                    if not is_valid:
                        errors.append(f"❌ {os.path.basename(file_path)}: {message}")
                        continue
                    
                    # Extract text from file
                    with open(file_path, 'rb') as f:
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
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            margin: 10px 0;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # 📚 Document Q&A with RAG
        
        Upload your research documents (PDF, TXT, DOCX, DOC) and ask questions about their content. 
        The system uses **Retrieval-Augmented Generation (RAG)** to find relevant information and provide accurate answers with source excerpts.
        
        **Features:**
        - 📎 Multiple file upload support
        - 🔍 Semantic search through documents
        - 📝 Context-aware answers with excerpts
        - 🎯 Source attribution and relevance scoring
        
        **Supported formats:** PDF, TXT, DOCX, DOC (max 50MB per file)
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # File upload section
                gr.Markdown("### 📎 Upload Documents")
                files_input = gr.File(
                    label="Choose files to upload",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".docx", ".doc"],
                    height=150
                )
                
                # Question input section
                gr.Markdown("### ❓ Ask Your Question")
                question_input = gr.Textbox(
                    label="Enter your question",
                    placeholder="What would you like to know about the uploaded documents?",
                    lines=4,
                    max_lines=6
                )
                
                # Submit button
                submit_btn = gr.Button(
                    "🔍 Get Answer",
                    variant="primary",
                    size="lg",
                    scale=1
                )
                
                # Clear button
                clear_btn = gr.Button(
                    "🗑️ Clear All",
                    variant="secondary",
                    size="sm"
                )
                
            with gr.Column(scale=2):
                # Answer output
                gr.Markdown("### 💡 Answer")
                answer_output = gr.Markdown(
                    value="📋 **Ready to help!** Upload documents and ask a question to get started.",
                    height=400,
                    show_label=False
                )
                
        # Sources and errors in expandable sections
        with gr.Row():
            with gr.Column():
                with gr.Accordion("📚 Sources & Context", open=False) as sources_accordion:
                    sources_output = gr.Markdown(
                        value="Source information will appear here after processing."
                    )
                
                with gr.Accordion("⚠️ Warnings & Errors", open=False, visible=False) as errors_accordion:
                    error_output = gr.Markdown()
        
        # Handle form submission
        def process_and_display(files, question):
            if not files or not question.strip():
                return (
                    "📋 **Instructions:**\n\n1. Upload one or more documents using the file upload area\n2. Enter your question in the text box\n3. Click 'Get Answer' to receive AI-powered insights\n\n*Both files and question are required.*",
                    "Upload documents and ask a question to see source information.",
                    "",
                    gr.update(visible=False)
                )
            
            # Show processing message
            processing_msg = f"🔄 **Processing {len(files)} file(s)...**\n\nPlease wait while I:\n- Extract text from documents\n- Create semantic chunks\n- Generate embeddings\n- Search for relevant content\n- Generate your answer"
            
            # Get file paths
            file_paths = [f.name for f in files] if files else []
            
            # Process documents and get answer
            answer, sources, errors = DocumentProcessor.process_documents_and_answer(file_paths, question)
            
            # Prepare outputs
            if answer:
                formatted_answer = f"## 💡 Answer\n\n{answer}"
                sources_info = sources if sources else "No source information available."
                errors_visible = bool(errors.strip())
            else:
                formatted_answer = f"❌ **Unable to generate answer**\n\n{errors}" if errors else "❌ No answer could be generated."
                sources_info = "No sources available due to processing error."
                errors_visible = bool(errors.strip())
            
            return (
                formatted_answer,
                sources_info,
                errors if errors else "",
                gr.update(visible=errors_visible)
            )
        
        def clear_all():
            return (
                None,  # Clear files
                "",    # Clear question
                "📋 **Ready to help!** Upload documents and ask a question to get started.",
                "Source information will appear here after processing.",
                "",
                gr.update(visible=False)
            )
        
        # Connect the submit button
        submit_btn.click(
            fn=process_and_display,
            inputs=[files_input, question_input],
            outputs=[answer_output, sources_output, error_output, errors_accordion]
        )
        
        # Connect clear button
        clear_btn.click(
            fn=clear_all,
            outputs=[files_input, question_input, answer_output, sources_output, error_output, errors_accordion]
        )
        
        # Also allow Enter key in question box
        question_input.submit(
            fn=process_and_display,
            inputs=[files_input, question_input],
            outputs=[answer_output, sources_output, error_output, errors_accordion]
        )
        
        # Example section
        gr.Markdown("""
        ---
        ## 📖 Example Questions
        
        Here are some effective question types to try:
        
        **📊 Analysis & Summary:**
        - "What are the main findings in these research papers?"
        - "Can you summarize the key methodologies used?"
        - "What are the primary conclusions across all documents?"
        
        **🔍 Specific Information:**
        - "What does the research say about [specific topic]?"
        - "What methodology was used in the [specific study]?"
        - "What are the limitations mentioned in the studies?"
        
        **💡 Insights & Recommendations:**
        - "What recommendations are provided?"
        - "What future research directions are suggested?"
        - "What are the practical applications mentioned?"
        
        **📈 Comparative Analysis:**
        - "How do the different papers compare on [topic]?"
        - "What contradictions exist between the documents?"
        - "What common themes emerge across all papers?"
        """)
    
    return demo


if __name__ == "__main__":
    # Create and launch the interface
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
        show_error=True,
        show_tips=True
    )