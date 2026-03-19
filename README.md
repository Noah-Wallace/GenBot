---
title: GenBot - Document Q&A Assistant
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
---

# 🧠 GenAI Research Assistant

> ⚕️ An intelligent, full-stack PDF research assistant that lets you query multiple research papers using natural language and get precise, context-aware answers powered by LLMs.

---

### 📊 Architecture Overview

```
                    ┌────────────┐
                    │   👤 User   │
                    └─────┬──────┘
                          │
        🗣️ Natural Language Question (via UI)
                          │
                    ┌─────▼──────┐
                    │ Classifier │
                    └─────┬──────┘
                          │
           ┌──────────────▼──────────────┐
           │ Embedding-Based Search (FAISS)│
           │  ⤷ Sentence Transformers       │
           └──────────────┬──────────────┘
                          │
            ┌─────────────▼────────────┐
            │   Top-K Relevant Chunks   │
            └─────────────┬────────────┘
                          │
                  ┌───────▼────────┐
                  │ Prompt Composer │
                  └───────┬────────┘
                          │
                  ┌───────▼──────────┐
                  │ Groq LLM (GPT-4) │
                  └───────┬──────────┘
                          │
                     🧾 Final Answer
                          │
                    ┌─────▼──────┐
                    │ 📄 UI Render │
                    └────────────┘
```

---

### 🚀 Features

* ✅ Upload and parse **multiple PDFs** (supports `.pdf`, `.docx`, `.txt`)
* 🧠 Ask **natural language** questions
* 🔍 Semantic search via **ChromaDB + Sentence Transformers**
* 🤖 Response generation using **Groq's GPT-4 class model**
* 🌙 Clean, dark-mode enabled frontend with smooth animations
* 🧾 Print answers or 📋 copy with one click
* 🛠️ Robust error handling and validations

---

### 🧰 Tech Stack

| Layer          | Tools/Frameworks                                  |
| -------------- | ------------------------------------------------- |
| 👩‍🎨 Frontend | HTML5, CSS3, Vanilla JS, Google Fonts, Coolors    |
| 🧠 LLM Backend | Groq API (OpenAI-compatible), Flask               |
| 🔎 Search      | ChromaDB, `sentence-transformers` (all-MiniLM-L6-v2) |
| 📄 Parsing     | PyMuPDF (fitz), python-docx                       |
| ☁️ Deployment  | Render.com                                        |

---

### 📁 Project Structure

```
GenAI-Healthcare/
├── app.py                  # Flask server
├── utils.py                # Core logic: parsing, embedding, querying
├── templates/              # Jinja2 templates (HTML pages)
│   ├── index.html
│   └── result.html
├── static/                 # CSS and static assets
│   └── styles.css
├── uploads/                # Temporary file uploads
├── requirements.txt        # Dependencies
├── render.yaml             # Render deployment config
├── README.md
```

---

### 🎨 Frontend Highlights

* ✨ Custom CSS using **Coolors** palette
* 🖋 **Abril Fatface** and **Segoe UI** typography
* 🌙 Toggleable **Dark Mode**
* 🔁 Smooth load/scroll animations
* 📋 One-click **copy to clipboard**
* 🖨️ Instant **print-ready formatting**
* 📱 Fully **mobile responsive**

---

### ⚙️ How It Works

1. User uploads multiple research papers (PDFs, docs)
2. Backend extracts text using PyMuPDF/docx parser
3. Chunks are created and embedded using `all-MiniLM-L6-v2`
4. ChromaDB finds most relevant text chunks
5. Context + question are passed to Groq’s GPT model
6. Answer is rendered on a clean, interactive UI

---

### ☁️ Deployment on Render

**1. `render.yaml`**

```yaml
services:
  - type: web
    name: genai-healthcare
    env: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "gunicorn app:app"
    envVars:
      - key: GROQ_API_KEY
        value: your-api-key-here
```

**2. `requirements.txt` (cleaned + optimized)**

```txt
Flask==3.1.1
numpy==2.3.1
Werkzeug==3.1.3
python-docx==1.2.0
PyMuPDF==1.26.3
groq==0.30.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
python-dotenv==1.0.1
gunicorn==21.2.0
```

**3. Procfile (if using instead of render.yaml):**

```txt
web: gunicorn app:app
```

---

### 💻 Run Locally

```bash
git clone https://github.com/yourusername/genai-healthcare-assistant.git
cd genai-healthcare-assistant

python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your-key-here" > .env

python app.py
```

Access at: [http://localhost:5000](http://localhost:5000)

---

### ✅ Future Plans

* 🧠 Support LLM streaming responses
* 📊 Add answer confidence and source citations
* 🔐 User auth and saved query history
* 📝 Export answers as downloadable PDF
* ⚛️ React + Flask hybrid upgrade


---

### 🤝 Contributing

Pull requests welcome! For major changes, open an issue first.

---

### 📄 License

**MIT** © 2025 Ayesha Tariq

---

### 💡 Inspiration

Inspired by ChatGPT, LangChain, and Groq’s blazing-fast LLMs — built completely from scratch for learning and showcasing AI + frontend development skills.

---


