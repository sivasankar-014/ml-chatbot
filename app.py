from flask import Flask, request, jsonify
import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

app = Flask(__name__)

DOCUMENT_PATH = "Machine Learning Documentation.docx"
CHROMA_DB_PATH = "./ML_chroma_db"

loader = Docx2txtLoader(DOCUMENT_PATH)
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if not os.path.exists(CHROMA_DB_PATH):
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_PATH
    )
else:
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embeddings
    )

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.3
)

@app.route("/")
def home():
    return "Machine Learning RAG Chatbot Running"

@app.route("/chat", methods=["POST"])
def chat():

    data = request.get_json()

    question = data["question"]

    docs = retriever.invoke(question)

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    prompt = f"""
You are a Machine Learning Knowledge Assistant.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return jsonify({
        "answer": response.content
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)