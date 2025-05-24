import os
import glob
import argparse
import chromadb
import pdfplumber
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.schema import Document

# Constants
BASE_DIR = "./data"
CHROMA_DB_DIR = "./chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Custom PDF loader using pdfplumber
def load_pdf_with_plumber(file_path):
    documents = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                documents.append(Document(page_content=text, metadata={"source_path": file_path}))
    return documents

# Step 1: Ingest and tag documents
def ingest_documents():
    client = chromadb.Client()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="india_china_docs", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)

    for country in ["India", "China"]:
        folder_path = os.path.join(BASE_DIR, country)
        for file_path in glob.glob(f"{folder_path}/*.pdf"):
            docs = load_pdf_with_plumber(file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            print(f"Ingesting {len(chunks)} chunks from {file_path} tagged as {country}")
            # Add country tag to metadata
            for chunk in chunks:
                chunk.metadata["source"] = country
            vectorstore.add_documents(chunks)
    # vectorstore.persist()

# Step 2: Query based on country tag
def query_documents(scope):
    user_question = input("Enter your question: ").strip()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(collection_name="india_china_docs", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
    llm = OllamaLLM(model="llama3.2:1b")
    print("Using LLM:", llm.model)
    print("Scope for query:", scope)
    print("Retrieving documents...")
    # Validate scope input

    if scope.lower() == "india":
        filters = {"source": "India"}
        docs = vectorstore.similarity_search(user_question, k=4, filter=filters)
    elif scope.lower() == "china":
        filters = {"source": "China"}
        docs = vectorstore.similarity_search(user_question, k=4, filter=filters)
    elif scope.lower() == "india+china":
        docs_india = vectorstore.similarity_search(user_question, k=2, filter={"source": "India"})
        docs_china = vectorstore.similarity_search(user_question, k=2, filter={"source": "China"})
        docs = docs_india + docs_china
    else:
        raise ValueError("Invalid scope. Use 'India', 'China', or 'India+China'")

    combined = "\n \n \n".join([doc.page_content for doc in docs])
    print("\nCOntext to the MODEL:\n", combined)
    
    response = llm.invoke(combined + f"\n\nAnswer the question: {user_question}")
    print("***************************************************************************")
    print("\nResponse:\n", response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ingest", "query"], required=True)
    parser.add_argument("--scope", choices=["India", "China", "India+China"], help="Scope for querying")
    args = parser.parse_args()

    if args.mode == "ingest":
        ingest_documents()
    elif args.mode == "query":
        if not args.scope:
            raise ValueError("Scope must be provided in query mode")
        query_documents(args.scope)
