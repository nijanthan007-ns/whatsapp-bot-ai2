import os
import requests
import tempfile
from fastapi import FastAPI, Request
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# === Environment Variables ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-...")
INSTANCE_ID = "instance133623"
ULTRAMSG_TOKEN = "shnmtd393b5963kq"
ULTRAMSG_URL = f"https://api.ultramsg.com/{INSTANCE_ID}/messages/chat"

# === GitHub PDF Config ===
GITHUB_API_URL = "https://api.github.com/repos/nijanthan007-ns/whatsapp-bot-ai2/contents/docs"
RAW_BASE_URL = "https://raw.githubusercontent.com/nijanthan007-ns/whatsapp-bot-ai2/main/docs"

app = FastAPI()

# === Load PDFs from GitHub ===
def load_documents():
    response = requests.get(GITHUB_API_URL)
    try:
        files = response.json()
        if not isinstance(files, list):
            print("GitHub API returned unexpected response:", files)
            return []

        docs = []
        for file in files:
            if isinstance(file, dict) and file.get("name", "").endswith(".pdf"):
                pdf_url = f"{RAW_BASE_URL}/{file['name']}"
                print(f"Downloading: {pdf_url}")
                try:
                    r = requests.get(pdf_url)
                    r.raise_for_status()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                        f.write(r.content)
                        tmp_path = f.name

                    loader = PyPDFLoader(tmp_path)
                    docs.extend(loader.load())
                    os.remove(tmp_path)
                except Exception as e:
                    print(f"Failed to load {file['name']}: {e}")
        return docs
    except Exception as e:
        print("Error loading GitHub PDFs:", e)
        return []

# === Split, Embed, and Store ===
def get_vector_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No documents loaded from GitHub.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(texts, embeddings)

retriever = get_vector_store().as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
    retriever=retriever,
    return_source_documents=False
)

# === WhatsApp Webhook ===
@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    data = await request.json()
    message = data.get("message")
    sender = data.get("from")

    if not message or not sender:
        return {"status": "invalid request"}

    print(f"Received from {sender}: {message}")
    try:
        response = qa_chain.run(message)
    except Exception as e:
        response = f"Error answering your question: {e}"

    payload = {
        "token": ULTRAMSG_TOKEN,
        "to": sender,
        "body": response,
    }

    try:
        requests.post(ULTRAMSG_URL, data=payload)
    except Exception as e:
        print("Error sending message:", e)

    return {"status": "ok"}
