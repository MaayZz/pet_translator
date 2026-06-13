import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Configuration
DOSSIER_BDD = os.path.join(os.path.dirname(__file__), "chroma_db_data")

class RAGRetriever:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None
        self._init_db()

    def _init_db(self):
        """Initializes the ChromaDB if it exists on disk."""
        if os.path.exists(DOSSIER_BDD):
            self.vector_store = Chroma(
                persist_directory=DOSSIER_BDD,
                embedding_function=self.embeddings
            )

    def index_pdf(self, pdf_path):
        """Indexes a PDF file into the local ChromaDB."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        for page in pages:
            page.metadata["source"] = os.path.basename(pdf_path)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(pages)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=DOSSIER_BDD
        )
        return len(chunks)

    def retrieve_context(self, query, top_k=3):
        """Retrieves top_k context chunks relevant to the query."""
        if self.vector_store is None:
            return "No RAG database available. Please index a PDF document about pet behavior first."
            
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k}
        )
        
        docs = retriever.invoke(query)
        if not docs:
            return "No relevant behavioral context found."
            
        context_text = "\n\n".join([f"[Source: {doc.metadata.get('source', '?')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}" for doc in docs])
        return context_text

if __name__ == "__main__":
    # Quick test
    rag = RAGRetriever()
    print("RAG Retriever initialized.")
