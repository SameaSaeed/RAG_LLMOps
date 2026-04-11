import os
import bs4
import cassio
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. NEW MODERN IMPORTS (Replacing langchain_classic)
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
## Connection to ASTRA DB

astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
astra_db_id = os.getenv("ASTRA_DB_ID")
cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

# Load Data
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("post-title", "post-content", "post-header")
    ))
)
text_documents = loader.load()

# Split Data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(text_documents)

# Vector Store Setup
embeddings = HuggingFaceEmbeddings()
astra_vector_store = Cassandra(
    embedding=embeddings,
    table_name="qa_mini_demo",
    session=None,
    keyspace=None
)
astra_vector_store.add_documents(docs)
retriever = astra_vector_store.as_retriever()

# LLM Setup
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

### --- CONVERSATIONAL RAG LOGIC --- ###

# 1. Rephrase Prompt: Handles context-aware questions (e.g., "Tell me more about that")
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create a retriever that understands history
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# 2. Answer Prompt: The final response generation
qa_system_prompt = (
    "You are an expert assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Think step by step before providing a detailed answer.\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 3. Combine everything into the final chain
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### --- EXECUTION --- ###

# Manage chat history manually in a list
chat_history = []

# Turn 1
input_1 = "Chain of thought (CoT; Wei et al. 2022) has become a standard prompting technique"
response1 = rag_chain.invoke({"input": input_1, "chat_history": chat_history})
print(f"AI: {response1['answer']}\n")

# Update history for the next turn
chat_history.extend([
    HumanMessage(content=input_1),
    AIMessage(content=response1["answer"]),
])

# Turn 2 (Testing the "Memory")
input_2 = "Who were the authors of that paper?"
response2 = rag_chain.invoke({"input": input_2, "chat_history": chat_history})
print(f"AI: {response2['answer']}")