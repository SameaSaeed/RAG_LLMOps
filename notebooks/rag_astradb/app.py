
import streamlit as st
import bs4
import time
import cassio
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.cassandra import Cassandra
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
import os


st.set_page_config(page_title="AstraDB RAG Demo", layout="wide")

# Sidebar for API Keys
with st.sidebar:
    st.title("Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password")
    astra_db_application_token = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
    astra_db_id = os.getenv("ASTRA_DB_ID")

st.title("🚀 AstraDB + Groq RAG")

if not groq_api_key:
    st.info("Please enter your Groq API Key in the sidebar to continue.")
    st.stop()

# Initialize Vector Store
if 'vector_store' not in st.session_state:
    with st.spinner("Initializing Astra DB..."):
        cassio.init(token=astra_token, database_id=astra_id)
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io",),
            bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_=("post-title","post-content","post-header")))
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_docs = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        st.session_state.vector_store = Cassandra(
            embedding=embeddings,
            table_name="qa_mini_demo_streamlit",
            session=None,
            keyspace=None
        )
        st.session_state.vector_store.add_documents(final_docs)

# Chain Setup
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
<context>{context}</context>
Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vector_store.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("Ask about AI Agents:")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_prompt})
    st.markdown(f"### Answer\n{response['answer']}")
    st.caption(f"Response time: {time.process_time() - start:.2f}s")
