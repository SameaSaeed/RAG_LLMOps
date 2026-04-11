from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configs.logger import GLOBAL_LOGGER as log
from configs.exceptions import DocumentPortalException
from utils.model_loader import ModelLoader
from utils.docs_loader import load_documents, FastAPIFileAdapter
import uuid
from datetime import datetime
import os


# -----------------------------
# Session helper
# -----------------------------
def generate_session_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"session_{timestamp}_{unique_id}"


# -----------------------------
# Chat Ingestor for RAG
# -----------------------------
class ChatIngestor:
    def __init__(self, temp_base: str = "data", session_id: Optional[str] = None):
        try:
            self.model_loader = ModelLoader()
            self.session_id = session_id or generate_session_id()
            self.temp_dir = Path(temp_base) / self.session_id
            self.temp_dir.mkdir(parents=True, exist_ok=True)

            # AstraDB init
            import cassio
            from langchain_community.vectorstores.cassandra import Cassandra

            cassio.init(
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
                database_id=os.getenv("ASTRA_DB_ID")
            )

            self.embeddings = self.model_loader.load_embeddings()
            self.vector_store = Cassandra(
                embedding=self.embeddings,
                table_name="qa_mini_demo",
                session=None,
                keyspace=None,
            )

            log.info("ChatIngestor initialized", session_id=self.session_id)

        except Exception as e:
            raise DocumentPortalException("Initialization error", e)

    # -------------------------
    # Split documents into chunks
    # -------------------------
    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        for c in chunks:
            c.metadata["session_id"] = self.session_id
        return chunks

    # -------------------------
    # Build retriever
    # -------------------------
    def build_retriever(
        self,
        uploaded_files: Iterable[FastAPIFileAdapter],
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5,
    ):
        try:
            # Save files
            paths = [uf.to_path(self.temp_dir) for uf in uploaded_files]
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No documents loaded")

            # Split and upsert
            chunks = self._split(docs, chunk_size, chunk_overlap)
            self.vector_store.add_documents(chunks)

            retriever = self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": 20,
                    "lambda_mult": 0.5,
                    "filter": {"session_id": self.session_id},
                }
            )
            return retriever

        except Exception as e:
            raise DocumentPortalException("Retriever build failed", e)
