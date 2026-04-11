from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from .logger import CustomLogger
from .exceptions import DocumentPortalException

# -------------------------
# 1. Setup structured logger
# -------------------------
logger = CustomLogger(log_dir="logs").get_logger("rag_pipeline")


# -------------------------
# 2. Pydantic models (simplified imports)
# -------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


class SourceDocument(BaseModel):
    source: str | None = None
    page: int | None = None
    score: float | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDocument] | None = None
    metadata: dict | None = None


# -------------------------
# 3. Prompts
# -------------------------
contextualize_question_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Given a conversation history and the most recent user query, rewrite the query "
        "as a standalone question that makes sense without relying on previous context. "
        "Do not provide an answer—only reformulate the question if necessary."
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

context_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are an assistant designed to answer questions using the provided context. "
        "Rely only on the retrieved information to form your response. "
        "If the answer is not found in the context, respond with 'I don't know.' "
        "Keep your answer concise and no longer than three sentences.\n\n{context}"
    )),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

PROMPT_REGISTRY = {
    "contextualize_question": contextualize_question_prompt,
    "context_qa": context_qa_prompt,
}


# -------------------------
# 4. Helper functions
# -------------------------
def reformulate_query(user_message: str, chat_history: List[HumanMessage | AIMessage]) -> List[HumanMessage | AIMessage]:
    """Generate standalone query from multi-turn chat history."""
    try:
        messages = PROMPT_REGISTRY["contextualize_question"].format_prompt(
            input=user_message,
            chat_history=chat_history
        ).to_messages()
        logger.info("Reformulated query", user_message=user_message, messages=[m.content for m in messages])
        return messages
    except Exception as e:
        raise DocumentPortalException("Failed to reformulate query", e, context={"user_message": user_message})


def answer_with_context(user_message: str, chat_history: List[HumanMessage | AIMessage], context: str) -> ChatResponse:
    """Simulate context-aware RAG answer."""
    try:
        PROMPT_REGISTRY["context_qa"].format_prompt(
            input=user_message,
            chat_history=chat_history,
            context=context
        ).to_messages()

        # Here you would send messages to LLM (Groq Llama3.3-70B) and get the response
        model_answer = "Simulated RAG answer based on retrieved context."

        logger.info(
            "Generated answer",
            user_message=user_message,
            answer=model_answer,
            context=context
        )

        # Example: populate sources and metadata
        sources = [SourceDocument(source="https://lilianweng.github.io/posts/2023-06-23-agent/", score=0.92)]
        metadata = {"top_k": 5, "retrieved_docs": len(sources)}

        return ChatResponse(answer=model_answer, sources=sources, metadata=metadata)
    except Exception as e:
        raise DocumentPortalException("Failed to generate context-aware answer", e, context={"user_message": user_message})


# -------------------------
# 5. Example pipeline usage
# -------------------------
if __name__ == "__main__":
    chat_history: List[HumanMessage | AIMessage] = []

    # Turn 1
    user_input_1 = "Hello there, I want to study about RAG"
    reformulated_msgs = reformulate_query(user_input_1, chat_history)
    chat_history.append(HumanMessage(content=user_input_1))
    chat_history.append(AIMessage(content=""))  # placeholder for answer

    # Turn 2
    user_input_2 = "What is the full form of it?"
    reformulated_msgs = reformulate_query(user_input_2, chat_history)
    chat_history.append(HumanMessage(content=user_input_2))
    chat_history.append(AIMessage(content=""))  # placeholder

    # Simulate context retrieval from AstraDB embeddings
    retrieved_context = "RAG stands for Retrieval-Augmented Generation."
    response = answer_with_context(user_input_2, chat_history, retrieved_context)
    print(response.json(indent=2))
