import os
import requests
import tempfile
from bs4 import BeautifulSoup
from fastapi import FastAPI, Request
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

app = FastAPI()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
GITHUB_DOCS_URL = "https://raw.githubusercontent.com/nijanthan007-ns/whatsapp-bot-ai2/main/docs/"
GITHUB_HTML_URL = "https://github.com/nijanthan007-ns/whatsapp-bot-ai2/tree/main/docs"

def get_all_pdf_filenames_from_github():
    r = requests.get(GITHUB_HTML_URL)
    soup = BeautifulSoup(r.text, "html.parser")
    pdfs = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf"):
            filename = href.split("/")[-1]
            pdfs.append(filename)

    return pdfs

def load_documents():
    docs = []
    filenames = get_all_pdf_filenames_from_github()
    for filename in filenames:
        url = f"{GITHUB_DOCS_URL}{filename}"
        try:
            r = requests.get(url)
            if r.status_code == 200 and len(r.content) > 1000:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(r.content)
                    loader = PyPDFLoader(tmp.name)
                    docs.extend(loader.load())
            else:
                print(f"Skipped empty or missing file: {filename}")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    return docs

def get_vector_store():
    docs = load_documents()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    return FAISS.from_documents(texts, embeddings)

retriever = get_vector_store().as_retriever()
chat_model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    message = data["message"].strip()
    sender = data["sender"]

    qa = RetrievalQA.from_chain_type(llm=chat_model, retriever=retriever)
    answer = qa.run(message)

    payload = {
        "token": os.environ["ULTRAMSG_TOKEN"],
        "to": sender,
        "body": answer
    }

    requests.post(
        f"https://api.ultramsg.com/{os.environ['ULTRAMSG_INSTANCE_ID']}/messages/chat",
        data=payload
    )

    return {"status": "success"}
