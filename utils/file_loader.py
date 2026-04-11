from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Optional
from configs.logger import CustomLogger
from configs.exceptions import DocumentPortalException

SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".txt", ".pptx", ".md",
    ".csv", ".xlsx", ".xls", ".db", ".sqlite", ".sqlite3"
}

# Local logger instance
log = CustomLogger().get_logger(__name__)


def save_uploaded_files(
    uploaded_files: Iterable,
    target_dir: Path,
    session_id: Optional[str] = None   # 👈 NEW (for Astra multi-user)
) -> List[Path]:
    """
    Save uploaded files and return local paths.
    Compatible with Astra RAG ingestion pipeline.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for uf in uploaded_files:
            name = getattr(uf, "filename", getattr(uf, "name", "file"))
            ext = Path(name).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=name)
                continue

            # Clean filename
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(name).stem).lower()

            # 🔥 Unique filename (kept your logic but fixed overwrite bug)
            unique_id = uuid.uuid4().hex[:8]
            fname = f"{safe_name}_{unique_id}{ext}"

            out = target_dir / fname

            # Write file safely
            with open(out, "wb") as f:
                if hasattr(uf, "file") and hasattr(uf.file, "read"):
                    uf.file.seek(0)
                    f.write(uf.file.read())

                elif hasattr(uf, "read"):
                    data = uf.read()
                    if isinstance(data, memoryview):
                        data = data.tobytes()
                    f.write(data)

                else:
                    buf = getattr(uf, "getbuffer", None)
                    if callable(buf):
                        data = buf()
                        if isinstance(data, memoryview):
                            data = data.tobytes()
                        f.write(data)
                    else:
                        raise ValueError("Unsupported uploaded file object")

            saved.append(out)

            # 🔥 Structured logging (Astra-friendly)
            log.info(
                "File saved for ingestion",
                uploaded=name,
                saved_as=str(out),
                session_id=session_id
            )

        log.info(
            "All files saved",
            total_files=len(saved),
            session_id=session_id
        )

        return saved

    except Exception as e:
        log.error(
            "Failed to save uploaded files",
            error=str(e),
            dir=str(target_dir),
            session_id=session_id
        )
        raise DocumentPortalException(
            "Failed to save uploaded files",
            e,
            context={"stage": "file_save", "session_id": session_id}
        ) from e
