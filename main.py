import os
import requests
from fastapi import FastAPI, Request
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN") or "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"
GITHUB_REPO = "nijanthan007-ns/whatsapp-bot-ai2"
GITHUB_DOCS_URL = f"https://api.github.com/repos/{GITHUB_REPO}/contents/docs"

app = FastAPI()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

def load_documents():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    response = requests.get(GITHUB_DOCS_URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"GitHub API error: {response.json()}")

    files = response.json()
    documents = []

    for file in files:
        if file["name"].endswith(".pdf"):
            pdf_url = file["download_url"]
            r = requests.get(pdf_url)
            local_path = f"/tmp/{file['name']}"
            with open(local_path, "wb") as f:
                f.write(r.content)
            loader = PyPDFLoader(local_path)
            documents.extend(loader.load())

    return documents

def get_vector_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No documents loaded from GitHub.")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(texts, embeddings)

retriever = get_vector_store().as_retriever()

@app.post("/")
async def chat(request: Request):
    body = await request.json()
    query = body.get("message", "")
    if not query:
        return {"error": "No message provided."}
    result = retriever.invoke(query)
    return {"response": result.page_content}
