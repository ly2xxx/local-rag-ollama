import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

try:
    import agent
    from agent import SmolAgentHelper
except ImportError:
    SmolAgentHelper = None

st.set_page_config(page_title="ChatPDF & SmolAgent")


def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            agent_text = st.session_state["assistant"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))


def save_agent_files():
    if "agent_uploaded_files" not in st.session_state:
        st.session_state["agent_uploaded_files"] = []

    for file in st.session_state.get("agent_file_uploader", []):
        file_path = os.path.join(tempfile.gettempdir(), file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        if not any(fpath == file_path for _, fpath in st.session_state["agent_uploaded_files"]):
            st.session_state["agent_uploaded_files"].append((file.name, file_path))


def process_agent_input():
    if st.session_state["agent_user_input"] and len(st.session_state["agent_user_input"].strip()) > 0:
        user_text = st.session_state["agent_user_input"].strip()

        # Build prompt hint if PDF files are uploaded for the agent
        uploaded_info = ""
        if st.session_state.get("agent_uploaded_files"):
            file_list = ", ".join([f"'{name}' at path '{path}'" for name, path in st.session_state["agent_uploaded_files"]])
            uploaded_info = f"[Available uploaded files to read using your read_pdf or read_file tool: {file_list}]\n\n"

        prompt_with_context = uploaded_info + user_text

        with st.session_state["agent_thinking_spinner"], st.spinner("Agent Thinking..."):
            try:
                # Ensure fresh helper with latest tools
                helper = SmolAgentHelper()
                response = helper.ask(prompt_with_context)
            except Exception as e:
                response = f"Error running agent: {e}"

        st.session_state["agent_messages"].append((user_text, True))
        st.session_state["agent_messages"].append((response, False))


def display_agent_messages():
    st.subheader("Agent Chat")
    for i, (msg, is_user) in enumerate(st.session_state["agent_messages"]):
        message(msg, is_user=is_user, key=f"agent_{i}")
    st.session_state["agent_thinking_spinner"] = st.empty()


def read_and_save_file():
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    if "assistant" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    if "agent_uploaded_files" not in st.session_state:
        st.session_state["agent_uploaded_files"] = []

    st.header("ChatPDF & SmolAgent")

    tab_rag, tab_agent = st.tabs(["RAG", "AGENT"])

    with tab_rag:
        st.subheader("Upload a document")
        st.file_uploader(
            "Upload document",
            type=["pdf"],
            key="file_uploader",
            on_change=read_and_save_file,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        st.session_state["ingestion_spinner"] = st.empty()

        display_messages()
        st.text_input("Message", key="user_input", on_change=process_input)

    with tab_agent:
        st.subheader("Upload PDF Document for Agent")
        st.file_uploader(
            "Upload document for Agent",
            type=["pdf", "txt"],
            key="agent_file_uploader",
            on_change=save_agent_files,
            label_visibility="collapsed",
            accept_multiple_files=True,
        )

        if st.session_state.get("agent_uploaded_files"):
            st.caption("Available files for Agent:")
            for name, fpath in st.session_state["agent_uploaded_files"]:
                st.text(f"📄 {name} ({fpath})")

        st.subheader("SmolAgents + LiteLLM (Ollama)")
        if SmolAgentHelper is not None:
            display_agent_messages()
            st.text_input("Ask Agent", key="agent_user_input", on_change=process_agent_input)
        else:
            st.info("Loading smolagents & litellm...")


if __name__ == "__main__":
    page()
