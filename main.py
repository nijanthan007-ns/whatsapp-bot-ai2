import os
import requests
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

app = FastAPI()

# === ENVIRONMENT VARIABLES ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")  # store as secret in Render
REPO = "nijanthan007-ns/whatsapp-bot-ai2"
FOLDER = "docs"

# === LOAD DOCUMENTS FROM GITHUB ===
def load_documents():
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}
    url = f"https://api.github.com/repos/{REPO}/contents/{FOLDER}"
    response = requests.get(url, headers=headers)
    files = response.json()

    documents = []
    for file in files:
        if file["name"].endswith(".pdf"):
            download_url = file["download_url"]
            file_path = f"/tmp/{file['name']}"
            with open(file_path, "wb") as f:
                f.write(requests.get(download_url).content)

            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents

# === VECTOR STORE ===
def get_vector_store():
    documents = load_documents()
    if not documents:
        raise ValueError("No documents loaded from GitHub.")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(texts, embeddings)

retriever = get_vector_store().as_retriever()
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
qa_chain = load_qa_chain(llm, chain_type="stuff")

# === FASTAPI ROUTE ===
@app.post("/")
async def bot(request: Request):
    data = await request.json()
    msg = data.get("message", "")
    phone = data.get("sender", "")

    docs = retriever.get_relevant_documents(msg)
    answer = qa_chain.run(input_documents=docs, question=msg)

    # Send reply back via UltraMsg API
    send_ultramsg_reply(phone, answer)
    return {"status": "ok", "answer": answer}

def send_ultramsg_reply(to_number, message):
    instance_id = "instance133623"
    token = "shnmtd393b5963kq"
    url = f"https://api.ultramsg.com/{instance_id}/messages/chat"
    payload = {"to": to_number, "body": message}
    headers = {"Content-Type": "application/json"}

    requests.post(url, json=payload, headers=headers, params={"token": token})
