from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain_groq import ChatGroq
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from typing import List
from functools import lru_cache
from pinecone import Pinecone, ServerlessSpec
from google.colab import userdata
from langchain_classic.chains import RetrievalQA
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
import os
import tempfile


# Extract text from PDF files
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents

extracted_data = load_pdf_files("/content/sample_data")



def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs

minimal_docs = filter_to_minimal_docs(extracted_data)

# Split the documents into smaller chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk

texts_chunk = text_split(minimal_docs)
print(f"Number of chunks: {len(texts_chunk)}")


def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings

embedding = download_embeddings()


PINECONE_API_KEY = userdata.get("PINECONE_API_KEY")  # retrieves your key
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


def create_pinecone_retriever(
    texts_chunk=None,
    index_name="rag",
    k=3
):
    """
    Creates (or connects to) a Pinecone index and returns a retriever.
    If texts_chunk is provided, documents are indexed.
    """

    # --- Pinecone client ---
    pc = Pinecone()

    # --- Create index if it does not exist ---
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(index_name)

    # --- Embeddings ---
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- If documents provided → index them ---
    if texts_chunk is not None:
        Pinecone.from_documents(
            documents=texts_chunk,
            embedding=embedding,
            index_name=index_name
        )

    # --- Connect to existing index ---
    vectorstore = Pinecone(
        index=index,
        embedding_function=embedding.embed_query
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )

    return retriever



os.environ["GROQ_API_KEY"] = userdata.get("GROQ_API_KEY")
chatModel = ChatGroq(
    model="llama-3.1-8b-instant",  # updated supported model
    temperature=0
)

system_prompt = (
    "You are an Medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
def create_rag_chain(
    retriever,
    chat_model,
    prompt
):
    """
    Creates and returns a RAG chain using a retriever and LLM.
    """

    question_answer_chain = create_stuff_documents_chain(
        llm=chat_model,
        prompt=prompt
    )

    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )

    return rag_chain