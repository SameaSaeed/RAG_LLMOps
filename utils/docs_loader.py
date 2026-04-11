from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from fastapi import UploadFile
from configs.logger import GLOBAL_LOGGER as log
from configs.exceptions import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_documents(paths: Iterable[Path]) -> List[Document]:
    docs: List[Document] = []
    try:
        for p in paths:
            ext = p.suffix.lower()
            if ext == ".pdf":
                loader = PyPDFLoader(str(p))
            elif ext == ".docx":
                loader = Docx2txtLoader(str(p))
            elif ext == ".txt":
                loader = TextLoader(str(p), encoding="utf-8")
            else:
                log.warning("Unsupported extension skipped", path=str(p))
                continue

            loaded_docs = loader.load()

            # Add source metadata
            for d in loaded_docs:
                d.metadata["source"] = str(p.name)

            docs.extend(loaded_docs)
            log.info("Document loaded", file=str(p.name), chunks=len(loaded_docs))

        log.info("All documents loaded", total_chunks=len(docs))
        return docs

    except Exception as e:
        log.error("Failed loading documents", error=str(e))
        raise DocumentPortalException(
            "Error loading documents",
            e,
            context={"stage": "document_loading"}
        ) from e


class FastAPIFileAdapter:
    """Adapter for FastAPI UploadFile."""
    def __init__(self, uf: UploadFile):
        self._uf = uf
        self.name = uf.filename or "file"

    def getbuffer(self) -> bytes:
        self._uf.file.seek(0)
        return self._uf.file.read()

    def to_path(self, save_dir: Path) -> Path:
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            file_path = save_dir / self.name
            with open(file_path, "wb") as f:
                f.write(self.getbuffer())
            log.info("Uploaded file saved", path=str(file_path))
            return file_path
        except Exception as e:
            log.error("Failed saving uploaded file", file=self.name, error=str(e))
            raise DocumentPortalException(
                "Error saving uploaded file",
                e,
                context={"file": self.name, "stage": "file_save"}
            ) from e
