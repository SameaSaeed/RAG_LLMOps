from typing import List, Optional
from operator import itemgetter
import os

from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from utils.model_loader import ModelLoader
from configs.prompts import PROMPT_REGISTRY
from configs.models import PromptType
from configs.exceptions import DocumentPortalException


class ConversationalRAG:
    """
    Session-scoped RAG engine for chat-style interaction.
    Handles question reformulation, context-aware retrieval, and LLM response generation.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm = ModelLoader().load_llm()
        self.chain = None

        # Prompts
        self.contextualize_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXTUALIZE_QUESTION.value]
        self.qa_prompt: ChatPromptTemplate = PROMPT_REGISTRY[PromptType.CONTEXT_QA.value]

    def load_retriever(self, table_name: str, k: int = 5):
        """
        Initialize vectorstore and retriever for the session.
        """
        try:
            import cassio
            from langchain_community.vectorstores.cassandra import Cassandra

            cassio.init(
                token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
                database_id=os.getenv("ASTRA_DB_ID")
            )

            embeddings = ModelLoader().load_embeddings()

            vectorstore = Cassandra(
                embedding=embeddings,
                table_name=table_name,
            )

            self.retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": 20,
                    "lambda_mult": 0.5,
                    "filter": {"session_id": self.session_id}
                }
            )

            self._build_chain()

        except Exception as e:
            raise DocumentPortalException("Retriever load error", e)

    def _format_docs(self, docs: List):
        """
        Combine retrieved documents into a single string for LLM input.
        """
        return "\n\n".join(d.page_content for d in docs)

    def _build_chain(self):
        """
        Construct the LLM chain:
        1. Question rewriter
        2. Retriever + context formatting
        3. Context-aware QA prompt
        """
        question_rewriter = (
            {"input": itemgetter("input"), "chat_history": itemgetter("chat_history")}
            | self.contextualize_prompt
            | self.llm
            | StrOutputParser()
        )

        retrieve_docs = question_rewriter | self.retriever | self._format_docs

        self.chain = (
            {
                "context": retrieve_docs,
                "input": itemgetter("input"),
                "chat_history": itemgetter("chat_history"),
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )

    def invoke(self, user_input: str, chat_history: Optional[List[BaseMessage]] = None) -> str:
        """
        Run a RAG query and return the LLM-generated answer.
        """
        if not self.chain:
            raise DocumentPortalException("Chain not initialized")

        result = self.chain.invoke({
            "input": user_input,
            "chat_history": chat_history or []
        })
        return result or "no answer generated."
