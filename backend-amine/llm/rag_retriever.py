import os

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db_data")

class RAGRetriever:
    def __init__(self):
        self.vector_store = None
        self.embeddings = None
        self._init_db()

    def _init_db(self):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            from langchain_chroma import Chroma
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            if os.path.exists(CHROMA_DIR):
                self.vector_store = Chroma(
                    persist_directory=CHROMA_DIR,
                    embedding_function=self.embeddings,
                )
        except ImportError:
            pass

    def retrieve_context(self, query, top_k=3):
        if self.vector_store is None:
            return "No RAG database available. Please index a PDF document about pet behavior first."
        try:
            retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": top_k},
            )
            docs = retriever.invoke(query)
            if not docs:
                return "No relevant behavioral context found."
            return "\n\n".join([
                f"[Source: {doc.metadata.get('source', '?')}, Page {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in docs
            ])
        except Exception as e:
            return f"RAG retrieval failed: {e}"
