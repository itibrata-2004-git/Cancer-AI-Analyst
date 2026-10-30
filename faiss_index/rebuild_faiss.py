from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
FAISS_DIR = BASE_DIR / "faiss_index"

# Use the same embedding model as app/agent.py
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Find PDFs in the recovered project
pdf_files = list(DATA_DIR.rglob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(f"No PDF files found under: {DATA_DIR}")

print(f"Found {len(pdf_files)} PDF files.")

# Load the PDF contents
documents = []

for pdf_path in pdf_files:
    try:
        print(f"Reading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents.extend(loader.load())
    except Exception as exc:
        print(f"Could not read {pdf_path.name}: {exc}")

if not documents:
    raise RuntimeError("No PDF text could be extracted.")

# Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = splitter.split_documents(documents)

if not chunks:
    raise RuntimeError("No text chunks were generated.")

print(f"Prepared {len(chunks)} text chunks.")

# Build a new index from scratch in memory
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save the new index
FAISS_DIR.mkdir(parents=True, exist_ok=True)
vectorstore.save_local(str(FAISS_DIR))

print(f"\nSuccess! New FAISS index saved to:\n{FAISS_DIR}")
print("Expected files: index.faiss and index.pkl")