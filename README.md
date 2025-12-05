# Local RAG

A professional, locally hosted Retrieval-Augmented Generation (RAG) system built to securely interact with your private documents. This application leverages **LangChain** for document processing, **ChromaDB** for vector storage, **Ollama** for local LLM inference, and **Streamlit** for a user-friendly chat interface.

🔗 **Repository:** [https://github.com/AnthonyBlv/local_RAG](https://github.com/AnthonyBlv/local_RAG)

![Status](https://img.shields.io/badge/Status-Active-success)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Features

*   **Local & Private:** All data processing and AI inference happen on your machine. No data leaves your network.
*   **Multi-Format Support:** Natively supports `.pdf`, `.docx`, `.txt`, `.csv`, `.md`, and `.xlsx` files.
*   **Smart Context Awareness:** Automatically retrieves relevant document chunks to answer questions accurately.
*   **Session Management:** Create, save, load, and delete chat sessions with persistent history.
*   **Interactive UI:** Clean Streamlit interface with real-time streaming, source citations, and direct file access.
*   **Auto-Sync:** One-click synchronization to update the vector database when you add or remove files.

## 🛠️ Architecture

The project is modularized for maintainability:
*   `frontend.py`: The Streamlit-based user interface (Presentation Layer).
*   `vector_store.py`: Handles document loading, splitting, embedding, and ChromaDB management (Data Layer).
*   `chatbot_logic.py`: Manages the interaction between the RAG system and the Ollama API (Logic Layer).

## 📋 Prerequisites

*   **Python 3.10+** installed.
*   **Ollama** installed on your system. [Download Ollama](https://ollama.com/)

## 📦 Installation

1.  **Clone the Repository**
    ```
    git clone https://github.com/AnthonyBlv/local_RAG.git
    cd local_RAG
    ```

2.  **Install Dependencies**
    ```
    pip install -r requirements.txt
    ```

3.  **Prepare Your Data**
    *   Create a folder named `data` (or update `DATA_PATH` in `vector_store.py`).
    *   Place your documents (PDFs, Word Docs, Text, CSVs, etc.) inside this folder.

## ⚙️ Setup & Usage

### 1. Pull the AI Model (Crucial Step)
This application uses `gemma3:1b` by default. You **must** download this model to your local machine before running the app.
```
ollama pull gemma3:1b
```
*Tip: You can change the default model by modifying the `MODEL` variable in `chatbot_logic.py`.*

### 2. Start the Backend (Ollama)
Launch the Ollama server in a dedicated terminal window.
```
ollama serve
```

### 3. Launch the Application
Open a new terminal window in the project directory and run:
```
streamlit run frontend.py
```

### 4. Using the App
*   **First Run:** The app will automatically initialize the vector database and index your documents.
*   **Chat:** Type your questions in the chat bar.
*   **Sources:** The AI will cite sources for its answers.
*   **Sync:** If you add new files to your data folder, click **"🔄 Sync Vector Space"** in the sidebar to update the knowledge base instantly.




