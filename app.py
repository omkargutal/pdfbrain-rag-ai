"""
PDFBrain — Streamlit + Gemini + Chroma (RAG)
Upload a PDF → Ask questions → Get accurate answers
"""
import os
from pathlib import Path
import streamlit as st
import numpy as np
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

# ====================== CONFIG ======================

# configure the Streamlit page
st.set_page_config(page_title="PDFBrain", page_icon="🤖", layout="wide")

# main heading and description
st.title("🤖 PDFBrain")
st.caption("Upload PDF → Ask Questions → Get AI Answers (RAG)")

# load .env in project root (do not override already-set environment variables)
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

def _get_env_key(name: str):
    """Return the value for `name` from session state (if present) or the environment.

    This lets the user paste a key into the left sidebar for the current session
    without persisting it to disk. Use `.env` or shell exports for permanent usage.
    """
    val = None
    try:
        val = st.session_state.get(name)
    except Exception:
        val = None
    return val or os.getenv(name)

# ====================== SIDEBAR SETTINGS ======================

with st.sidebar:
    st.header("API Keys")
    gemini_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY2") or "",
        type="password",
    )
    save_env = st.checkbox("Save key to local .env file (overwrites existing)")
    if st.button("Apply keys"):
        if gemini_input:
            st.session_state["GEMINI_API_KEY2"] = gemini_input
            os.environ["GEMINI_API_KEY2"] = gemini_input

        if save_env:
            try:
                env_values = {}
                if env_path.exists():
                    for line in env_path.read_text().splitlines():
                        if "=" in line:
                            k, v = line.split("=", 1)
                            env_values[k] = v

                if gemini_input:
                    env_values["GEMINI_API_KEY2"] = gemini_input

                env_path.write_text("\n".join(f"{k}={v}" for k, v in env_values.items()))
                st.success("Saved key to .env (local file)")
            except Exception as e:
                st.error(f"Failed to write .env: {e}")

    st.markdown("---")

# retrieve the key after sidebar interaction
GEMINI_API_KEY = _get_env_key("GEMINI_API_KEY2")
if not GEMINI_API_KEY:
    st.error(
        "GEMINI_API_KEY not set. Add it to a local .env (ignored) or export it in your environment."
    )
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)

# simple in-memory storage structure will be kept in session state
# no need for an external vector database

# ====================== HELPERS ======================

def extract_text(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text, size=1200, overlap=200):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def embed_text(text):
    return client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    ).embeddings[0].values

# ====================== SIDEBAR ======================

with st.sidebar:
    st.header("📄 Upload Document")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_uploader")

    st.divider()
    st.header("📊 Token Usage")

    st.session_state.setdefault("input_tokens", 0)
    st.session_state.setdefault("output_tokens", 0)

    col1, col2 = st.columns(2)
    col1.metric("Input", st.session_state.input_tokens)
    col2.metric("Output", st.session_state.output_tokens)

    st.metric("Total", st.session_state.input_tokens + st.session_state.output_tokens)

    row = st.columns([1,1])
    if row[0].button("Reset Tokens"):
        st.session_state.input_tokens = 0
        st.session_state.output_tokens = 0
        st.rerun()
    if row[1].button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat = []
        st.rerun()

# ====================== PDF PROCESS ======================

if uploaded_file:
    if st.session_state.get("file") != uploaded_file.name:
        with st.spinner("Indexing document..."):
            text = extract_text(uploaded_file)
            chunks = chunk_text(text)

            # compute embeddings for each chunk and store in session
            data = []
            for chunk in chunks:
                data.append((chunk, embed_text(chunk)))

            st.session_state.file = uploaded_file.name
            st.session_state.data = data
            st.session_state.messages = []
            st.session_state.chat = []

        st.success(f"Indexed {len(chunks)} chunks")
        # show a small preview of the extracted text
        with st.expander("Preview extracted text"):
            preview = text if len(text) <= 1000 else text[:1000] + "..."
            st.write(preview)

# ====================== CHAT ======================

st.session_state.setdefault("messages", [])
st.session_state.setdefault("chat", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ====================== Q&A ======================

# once a file has been selected/processed we always allow questions
if uploaded_file:
    if prompt := st.chat_input("Ask your question..."):

        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        query_emb = embed_text(prompt)
        # compute cosine similarities against stored embeddings
        def cosine(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        sims = [cosine(query_emb, emb) for _, emb in st.session_state.data]
        top_idx = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:3]
        context = "\n\n".join(st.session_state.data[i][0] for i in top_idx)

        system_prompt = f"""
You are an AI assistant. Answer ONLY using the provided context.
If not found, say: "Not available in the document."

Context:
{context}
"""

        st.session_state.chat.append(
            types.Content(role="user", parts=[types.Part(text=prompt)])
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=st.session_state.chat,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt
                    )
                )

                answer = response.text

                if response.usage_metadata:
                    st.session_state.input_tokens += response.usage_metadata.prompt_token_count
                    st.session_state.output_tokens += response.usage_metadata.candidates_token_count

                st.markdown(answer)

        st.session_state.chat.append(
            types.Content(role="model", parts=[types.Part(text=answer)])
        )

        st.session_state.messages.append({"role": "assistant", "content": answer})

        if len(st.session_state.chat) > 20:
            st.session_state.chat = st.session_state.chat[-20:]

        st.rerun()

else:
    st.info("👈 Upload a PDF to start chatting. Once indexed, ask your questions using the chat box.")