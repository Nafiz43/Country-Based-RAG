import os
import glob
import argparse
import chromadb
import pdfplumber
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Constants
BASE_DIR = "./"
CHROMA_DB_DIR = "./chroma_db"
CHUNK_SIZE = 800
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

# Custom embedding class using instructor-large
class InstructorEmbeddings:
    def __init__(self, model_name="hkunlp/instructor-large"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, texts):
        prompts = [["Represent the document for retrieval: " + t] for t in texts]
        return self.model.encode(prompts, normalize_embeddings=True).tolist()

# Step 1: Ingest and tag documents
def ingest_documents():
    client = chromadb.Client()
    embeddings = InstructorEmbeddings()
    vectorstore = Chroma(collection_name="india_china_docs", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)

    for country in ["India", "China"]:
        folder_path = os.path.join(BASE_DIR, country)
        for file_path in glob.glob(f"{folder_path}/*.pdf"):
            docs = load_pdf_with_plumber(file_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            chunks = splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source"] = country
            vectorstore.add_documents(chunks)
    vectorstore.persist()

# Step 2: Query based on country tag
def query_documents(scope):
    embeddings = InstructorEmbeddings()
    vectorstore = Chroma(collection_name="india_china_docs", embedding_function=embeddings, persist_directory=CHROMA_DB_DIR)
    llm = Ollama(model="mistral")

    if scope.lower() == "india":
        filters = {"source": "India"}
        docs = vectorstore.similarity_search("India specific question", k=4, filter=filters)
    elif scope.lower() == "china":
        filters = {"source": "China"}
        docs = vectorstore.similarity_search("China specific question", k=4, filter=filters)
    elif scope.lower() == "india+china":
        docs_india = vectorstore.similarity_search("India specific question", k=2, filter={"source": "India"})
        docs_china = vectorstore.similarity_search("China specific question", k=2, filter={"source": "China"})
        docs = docs_india + docs_china
    else:
        raise ValueError("Invalid scope. Use 'India', 'China', or 'India+China'")

    combined = "\n".join([doc.page_content for doc in docs])
    response = llm.predict(combined + "\n\nAnswer the question based on the information above.")
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
