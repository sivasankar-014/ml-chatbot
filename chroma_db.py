import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_google_genai import GoogleGenerativeAIEmbeddings

# ==========================================
# CONFIG
# ==========================================
import os



DOCUMENT_PATH = "Machine Learning Documentation.docx"
CHROMA_DB_PATH = "./ML_chroma_db"

# ==========================================
# LOAD DOC
# ==========================================

loader = Docx2txtLoader(DOCUMENT_PATH)

documents = loader.load()

print("Documents Loaded:", len(documents))

# ==========================================
# SPLIT
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

print("Chunks:", len(chunks))

# ==========================================
# EMBEDDINGS
# ==========================================
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# ==========================================
# CREATE CHROMADB
# ==========================================

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH
)

print("ChromaDB Created Successfully")







