import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data():
    base_dir = os.path.dirname(os.path.dirname(__file__))
    file_path = os.path.join(base_dir, "data", "pet_behavior.txt")
    chroma_dir = os.path.join(os.path.dirname(__file__), "chroma_db_data")

    print(f"Loading {file_path}...")
    loader = TextLoader(file_path)
    documents = loader.load()

    print("Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    docs = text_splitter.split_documents(documents)

    for i, doc in enumerate(docs):
        doc.metadata["source"] = "Pet Behavior Guide 2026"
        doc.metadata["page"] = i + 1

    print("Generating embeddings and saving to Chroma...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Create or update vector store
    db = Chroma.from_documents(
        docs, 
        embeddings, 
        persist_directory=chroma_dir
    )
    
    print(f"Successfully ingested {len(docs)} chunks into {chroma_dir}")

if __name__ == "__main__":
    ingest_data()
