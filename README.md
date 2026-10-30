# 🧬 Cancer AI Analyst

An AI-powered application designed to help users explore and understand cancer-related research documents using Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and semantic search.

The project combines **Ollama, FAISS, LangChain, and Streamlit** to retrieve relevant information from cancer-related PDF documents and generate context-aware responses.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📌 Project Overview

Cancer research involves a large volume of scientific papers, reports, and medical documents. Finding relevant information in these resources can be time-consuming.

Cancer AI Analyst aims to simplify document exploration by allowing users to interact with cancer-related information through an AI-powered interface.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant passages from a collection of documents and use them as context for generating responses.

## ✨ Key Features

- **AI-Powered Question Answering:** Ask questions about cancer-related research documents.
- **PDF Document Processing:** Extract text from PDF documents for analysis.
- **Retrieval-Augmented Generation (RAG):** Generate responses using retrieved document context.
- **FAISS Vector Search:** Search for semantically relevant document chunks.
- **Local LLM Integration:** Use Ollama to run supported language models locally.
- **Semantic Embeddings:** Use the `nomic-embed-text` embedding model to represent document content.
- **Interactive Web Interface:** Explore information through a Streamlit application.
- **Local Document Indexing:** Build and save a searchable vector index from the available PDF collection.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| LangChain | Document processing and LLM integration |
| Ollama | Local language model and embedding model runtime |
| FAISS | Vector storage and similarity search |
| `nomic-embed-text` | Text embeddings |
| PyPDF | PDF document loading and text extraction |
| Recursive Character Text Splitter | Splitting documents into searchable chunks |

## 🏗️ How It Works

1. **Document Loading:** Cancer-related PDF files are loaded from the project's data directory.
2. **Text Extraction:** Text is extracted from the PDF documents.
3. **Text Chunking:** Extracted text is divided into smaller, overlapping chunks.
4. **Embedding Generation:** The embedding model converts the chunks into numerical representations.
5. **Vector Indexing:** FAISS stores the embeddings for similarity-based retrieval.
6. **Question Processing:** The user submits a question through the application.
7. **Context Retrieval:** Relevant document chunks are retrieved from the vector index.
8. **Response Generation:** The language model generates an answer using the retrieved context.

## 📂 Project Structure

The main project structure is expected to resemble the following:

```text
Cancer-AI-Analyst/
│
├── app/
│   ├── app.py
│   └── agent.py
│
├── data/
│   └── cancer-related PDF documents
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── models/
│   └── trained models, if applicable
│
├── notebooks/
│   └── analysis notebooks, if applicable
│
├── reports/
│   └── project reports, if applicable
│
├── rebuild_faiss.py
├── requirements.txt
├── .gitignore
└── README.md
```

*Note: The directory tree is illustrative. Keep only folders and files that actually exist in your repository. The `rebuild_faiss.py` script can be placed in the project root for easier maintenance.*

## ⚙️ Installation and Setup

### Prerequisites

Install the following before running the project:

- Python
- Git (optional, for cloning the repository)
- Ollama
- The required Ollama language model
- The `nomic-embed-text` embedding model

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/Cancer-AI-Analyst.git
cd Cancer-AI-Analyst
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a Virtual Environment

**Windows:**

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can run the commands from a suitable terminal or activate the environment using the method recommended for your Python installation.

### 3. Install Dependencies

If `requirements.txt` is located in the project root:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up Ollama

Install Ollama from its official website:

https://ollama.com/

Start Ollama, then download the embedding model:

```bash
ollama pull nomic-embed-text
```

Also download the language model required by your application. For example, if your code is configured to use a model named `llama3.2`, run:

```bash
ollama pull llama3.2
```

Use the language model specified in your project's code if it differs from this example.

### 5. Build the FAISS Index

If the repository includes the `rebuild_faiss.py` script, run it from the project root:

```bash
python rebuild_faiss.py
```

The script should process the PDF documents and create the FAISS index in the configured `faiss_index` directory.

If the index is already included and compatible with your environment, rebuilding may not be necessary.

### 6. Run the Application

From the project root, run:

```bash
streamlit run app/app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser to access the application.

## 💬 Example Use Cases

Depending on the documents included in the index, users may ask questions such as:

- What information do the available documents provide about a particular type of cancer?
- What risk factors are discussed in the indexed research?
- What symptoms or treatment approaches are described in the source documents?
- Which passages in the available documents are relevant to a research question?
- What are the key findings of a particular indexed research document?

**Important:** Answers depend on the quality, completeness, and relevance of the indexed documents. Verify important information against the original sources.

## 🔒 Privacy and Security

- Running supported models locally can reduce the need to send prompts to an external LLM service.
- Privacy also depends on the application's implementation, configured services, and network behavior.
- Do not commit API keys, passwords, access tokens, or private configuration files.
- Review datasets and documents before publishing them to ensure you have permission to redistribute them.
- Avoid uploading unnecessary virtual environments or temporary files.

## ⚠️ Medical Disclaimer

Cancer AI Analyst is an educational and research-support project. It is not a certified medical device and should not be used as the sole basis for clinical decisions.

The application may produce incomplete, inaccurate, or misleading information. Always consult qualified healthcare professionals for medical advice, diagnosis, and treatment decisions.

## 🚀 Future Improvements

Potential future enhancements include:

- Support for additional medical research document formats.
- Source references and document-level citations in generated answers.
- Improved retrieval and reranking of relevant passages.
- Better document upload and indexing workflows.
- Expanded evaluation of answer quality and retrieval accuracy.
- More comprehensive error handling and logging.
- User-friendly document management and search capabilities.

These are possible future improvements, not claims that they are already implemented.

## 👨‍💻 Author

**Itibrata Sahoo**

Computer Science Engineering | AI and Data Analytics

GitHub: https://github.com/itibrata-2004-git

Replace the placeholder with your actual GitHub profile URL before publishing.

## 📄 License

Choose a license that matches your intentions and the permissions associated with your project's code and data.

If you want others to reuse your code, consider adding an appropriate open-source license. Do not assume that a license covering your code also grants permission to redistribute medical papers or datasets.
