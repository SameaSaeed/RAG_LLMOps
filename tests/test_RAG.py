import os
import pytest
from unittest.mock import patch, MagicMock

from src.RAG import ConversationalRAG
from configs.exceptions import DocumentPortalException

# Mock cassio to avoid import issues
import sys
sys.modules['cassio'] = MagicMock()


# -------------------------
# Test: Error handling when RAG chain not initialized
# -------------------------
@patch.dict(os.environ, {"GROQ_API_KEY": "dummy"})
def test_invoke_raises_when_chain_not_initialized():
    rag = ConversationalRAG(session_id="test_session")
    # Chain not built yet
    rag.chain = None
    with pytest.raises(DocumentPortalException) as excinfo:
        rag.invoke("Hello")
    assert "Chain not initialized" in str(excinfo.value)


# -------------------------
# Test: Error when Astra credentials missing
# -------------------------
@patch.dict(os.environ, {"GROQ_API_KEY": "dummy"}, clear=True)
def test_load_retriever_raises_missing_credentials():
    rag = ConversationalRAG(session_id="test_session")
    with pytest.raises(DocumentPortalException) as excinfo:
        rag.load_retriever(table_name="qa_table")
    assert "Retriever load error" in str(excinfo.value)


# -------------------------
# Test: Load retriever with mocked Astra and LLM
# -------------------------
@patch.dict(os.environ, {"ASTRA_DB_APPLICATION_TOKEN": "fake_token", "ASTRA_DB_ID": "fake_db", "GROQ_API_KEY": "dummy"})
@patch("src.RAG.ModelLoader.load_embeddings")
@patch("cassio.init")
@patch("langchain_community.vectorstores.cassandra.Cassandra")
@patch.object(ConversationalRAG, '_build_chain')
def test_load_retriever_success(mock_build_chain, mock_cassandra, mock_cassio_init, mock_load_embeddings):
    mock_load_embeddings.return_value = "fake_embedding"
    mock_vector_store = MagicMock()
    mock_vector_store.as_retriever.return_value = "retriever_obj"
    mock_cassandra.return_value = mock_vector_store

    rag = ConversationalRAG(session_id="test_session")
    rag.load_retriever(table_name="qa_table", k=3)

    assert rag.retriever == "retriever_obj"
    mock_cassio_init.assert_called_once()
    mock_cassandra.assert_called_once_with(
        embedding="fake_embedding",
        table_name="qa_table",
    )
    mock_vector_store.as_retriever.assert_called_once()


# -------------------------
# Test: invoke returns "no answer generated." if chain returns None
# -------------------------
@patch.dict(os.environ, {"GROQ_API_KEY": "dummy"})
@patch("src.RAG.ModelLoader.load_llm")
def test_invoke_returns_default_when_chain_returns_none(mock_load_llm):
    # Mock LLM chain
    mock_llm = MagicMock()
    mock_load_llm.return_value = mock_llm

    rag = ConversationalRAG(session_id="test_session")
    rag.chain = MagicMock()
    rag.chain.invoke = MagicMock(return_value=None)  # simulate empty response

    result = rag.invoke("Hello")
    assert result == "no answer generated."
