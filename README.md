# PDFBrain

PDFBrain is a Streamlit-based FAQ bot that lets you upload PDF documents and chat with their content using Gemini and ChromaDB for retrieval-augmented generation (RAG).

## 🚀 Features

- 📄 **PDF Upload** — Upload any PDF document
- ✂️ **Auto Chunking** — Splits document into searchable pieces
- 🔍 **Semantic Search** — Finds relevant chunks using embeddings
- 💬 **Chat Interface** — Conversational Q&A with memory
- 📊 **Token Counter** — Track API usage in real-time

## 🛠 Tech Stack

- **Streamlit** — UI
- **Gemini** — Embeddings + LLM
- **ChromaDB** — Vector storage
- **PyPDF2** — PDF parsing

## 📁 Files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies

## ✅ Getting Started

1. **Clone the repository** (replace with your actual repo URL):
   ```bash
   git clone https://github.com/omkargutal/pdfbrain-rag-ai.git
   
   `cd filename`
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Set your Gemini API key in a `.env` file:
   ```env
   GEMINI_API_KEY2=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## 📄 Usage

- Upload a PDF via the sidebar.
- Wait for it to be indexed (you'll see a preview and chunk count).
- Ask questions in the chat box and get answers sourced from your document.

## 🧭 How to Use

1. **Open the Streamlit interface** after running the app. The sidebar will guide you through uploading your document.
2. **View document chunks** in the preview area to ensure the file was processed correctly.
3. **Interact via chat** by typing natural language questions and receive answers with source citations.
4. **Monitor token usage** using the counter in the header to manage API costs.

---

*Created with ❤️ using Streamlit, Gemini, and ChromaDB.*

---

*Created with ❤️ using Streamlit, Gemini, and ChromaDB.*