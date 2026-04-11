import streamlit as st
import requests

# ----------------------------
# Config
# ----------------------------
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Astra RAG Chat", layout="wide")

st.title(" Astra RAG Chat")

# ----------------------------
# Session State
# ----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------
# Upload Section
# ----------------------------
st.subheader("📄 Upload Documents")

uploaded_files = st.file_uploader(
    "Upload files",
    accept_multiple_files=True
)

if st.button("Upload & Index"):
    if not uploaded_files:
        st.warning("Upload at least one file")
    else:
        files = [
            ("files", (f.name, f.getvalue()))
            for f in uploaded_files
        ]

        with st.spinner("Uploading & indexing..."):
            res = requests.post(f"{API_URL}/upload", files=files)

        if res.status_code == 200:
            data = res.json()
            st.session_state.session_id = data["session_id"]
            st.session_state.chat_history = []
            st.success("✅ Documents indexed successfully")
        else:
            st.error(res.text)

# ----------------------------
# Chat Section
# ----------------------------
st.subheader("💬 Chat")

if not st.session_state.session_id:
    st.info("Upload documents first")
else:
    user_input = st.chat_input("Ask something...")

    if user_input:
        # Save user message
        st.session_state.chat_history.append(("user", user_input))

        payload = {
            "session_id": st.session_state.session_id,
            "message": user_input
        }

        with st.spinner("Thinking... 🤖"):
            res = requests.post(f"{API_URL}/chat", json=payload)

        if res.status_code == 200:
            answer = res.json()["answer"]
        else:
            answer = f"Error: {res.text}"

        # Save assistant response
        st.session_state.chat_history.append(("assistant", answer))

    # ----------------------------
    # Display Chat
    # ----------------------------
    for role, msg in st.session_state.chat_history:
        if role == "user":
            with st.chat_message("user"):
                st.markdown(msg)
        else:
            with st.chat_message("assistant"):
                st.markdown(msg)
