import os
import requests
from flask import Flask, request
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

OPENAI_API_KEY = "sk-proj-..."  # Replace with your real key
ULTRAMSG_INSTANCE_ID = "instance133623"
ULTRAMSG_TOKEN = "shnmtd393b5963kq"
GITHUB_DOCS_URL = "https://raw.githubusercontent.com/your-username/your-repo-name/main/docs/"  # Update this

app = Flask(__name__)
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_documents():
    import tempfile
    docs = []
    filenames = [
        "machine_manual1.pdf",
        "wiring_guide.pdf"
    ]
    for filename in filenames:
        url = f"{GITHUB_DOCS_URL}{filename}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            r = requests.get(url)
            tmp.write(r.content)
            loader = PyPDFLoader(tmp.name)
            docs.extend(loader.load())
    return docs

def get_vector_store():
    docs = load_documents()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    return vectordb

retriever = get_vector_store().as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(), retriever=retriever)

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.form
    msg = data.get("message")
    sender = data.get("from")

    if not msg:
        return "OK"

    answer = qa_chain.run(msg)
    send_message(sender, answer)
    return "OK"

def send_message(to, message):
    url = f"https://api.ultramsg.com/{ULTRAMSG_INSTANCE_ID}/messages/chat"
    payload = {"to": to, "body": message}
    headers = {"Content-Type": "application/json", "token": ULTRAMSG_TOKEN}
    requests.post(url, json=payload, headers=headers)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
