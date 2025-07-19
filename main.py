from fastapi import FastAPI, Request
import requests
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

OPENAI_API_KEY = "your-openai-api-key"

# GitHub API URL for your docs folder
GITHUB_API_URL = "https://api.github.com/repos/nijanthan007-ns/whatsapp-bot-ai2/contents/docs"
RAW_BASE_URL = "https://raw.githubusercontent.com/nijanthan007-ns/whatsapp-bot-ai2/main/docs"

def load_documents():
    response = requests.get(GITHUB_API_URL)
    files = response.json()

    docs = []
    for file in files:
        if file["name"].endswith(".pdf"):
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

def get_vector_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No documents loaded!")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(texts, embeddings)

retriever = get_vector_store().as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY), retriever=retriever)

@app.post("/")
async def handle_message(request: Request):
    data = await request.json()
    message = data.get("message")
    if not message:
        return {"error": "No message provided"}
    answer = qa_chain.run(message)
    return {"answer": answer}
