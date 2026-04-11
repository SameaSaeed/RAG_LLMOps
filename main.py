from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List, Dict

from src.index import ChatIngestor
from src.RAG import ConversationalRAG
from src.evaluation_api import router as evaluation_router

from langchain_core.messages import HumanMessage, AIMessage
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = FastAPI(
    title="LLMOps API",
    description="FastAPI backend for LLMOps with RAG and evaluation capabilities",
    version="1.0.0"
)

# Include evaluation API routes
app.include_router(evaluation_router)

SESSIONS: Dict[str, List[dict]] = {}


class FileAdapter:
    def __init__(self, f: UploadFile):
        self.file = f
        self.name = f.filename

    def getbuffer(self):
        self.file.file.seek(0)
        return self.file.file.read()

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    ingestor = ChatIngestor()
    session_id = ingestor.session_id

    wrapped = [FileAdapter(f) for f in files]

    ingestor.build_retriever(wrapped)

    SESSIONS[session_id] = []

    return {"session_id": session_id}


@app.post("/chat")
async def chat(session_id: str, message: str):
    if session_id not in SESSIONS:
        raise HTTPException(400, "Invalid session")

    rag = ConversationalRAG(session_id)
    rag.load_retriever("qa_mini_demo")

    history = []
    for m in SESSIONS[session_id]:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))

    answer = rag.invoke(message, history)

    SESSIONS[session_id].append({"role": "user", "content": message})
    SESSIONS[session_id].append({"role": "assistant", "content": answer})

    return {"answer": answer}
