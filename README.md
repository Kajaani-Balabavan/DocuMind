# 🤖 DocuMind - Intelligent Document Q&A System

A production-ready RAG (Retrieval-Augmented Generation) system built with Python, Streamlit, and Hugging Face APIs. Upload documents and ask intelligent questions powered by state-of-the-art NLP models.

## ✨ Features

- 📄 **Multi-format Support**: PDF, DOCX, TXT files
- 🔍 **Intelligent Search**: FAISS-powered vector similarity search
- 🤖 **AI-Powered Answers**: Hugging Face transformer models
- 💬 **Chat Interface**: Interactive Q&A experience
- 📊 **Response Analytics**: Confidence scoring and source tracking

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Vector Database**: FAISS
- **Embeddings**: SentenceTransformers
- **LLM**: Hugging Face Inference API (deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- **Document Processing**: PyPDF2, python-docx

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Kajaani-Balabavan/DocuMind.git
cd DocuMind
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
streamlit run app.py
```

### 4. Upload Documents & Start Chatting!

1. Upload your documents using the sidebar
2. Wait for processing to complete
3. Ask questions in the chat interface
