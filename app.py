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

try:
    import agentic_rag
    from agentic_rag import AgenticRAGHelper, render_agentic_rag_graph
except ImportError:
    AgenticRAGHelper = None
    render_agentic_rag_graph = None

st.set_page_config(page_title="Local RAG, SmolAgent & Agentic RAG", layout="wide")


# --- Tab 1: Classic RAG Helpers ---
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


# --- Tab 2: SmolAgent Helpers ---
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


# --- Tab 3: Agentic RAG Helpers ---
def ingest_agentic_rag_files():
    if "agentic_rag_helper" not in st.session_state or st.session_state["agentic_rag_helper"] is None:
        st.session_state["agentic_rag_helper"] = AgenticRAGHelper()

    for file in st.session_state.get("agentic_file_uploader", []):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["agentic_ingestion_spinner"], st.spinner(f"Ingesting into Agentic RAG: {file.name}"):
            res = st.session_state["agentic_rag_helper"].ingest_document(file_path)
            st.toast(res)
        os.remove(file_path)


def process_agentic_rag_input():
    if st.session_state["agentic_user_input"] and len(st.session_state["agentic_user_input"].strip()) > 0:
        user_text = st.session_state["agentic_user_input"].strip()
        thread_id = st.session_state.get("agentic_thread_id", "session_1")

        if "agentic_rag_helper" not in st.session_state or st.session_state["agentic_rag_helper"] is None:
            st.session_state["agentic_rag_helper"] = AgenticRAGHelper()

        with st.session_state["agentic_thinking_spinner"], st.spinner("Agentic RAG Core Reasoning (Thought -> Action -> Observation)..."):
            res = st.session_state["agentic_rag_helper"].ask(user_text, thread_id=thread_id)

        st.session_state["agentic_messages"].append({
            "user": user_text,
            "assistant": res.get("answer", "No response"),
            "scratchpad": res.get("scratchpad", []),
        })


def display_agentic_rag_messages():
    st.subheader("Agentic Multi-Turn Chat")
    for i, msg_dict in enumerate(st.session_state["agentic_messages"]):
        message(msg_dict["user"], is_user=True, key=f"agentic_u_{i}")
        message(msg_dict["assistant"], is_user=False, key=f"agentic_a_{i}")
        
        # Display Agent Scratchpad (Thought -> Action -> Observation)
        scratchpad = msg_dict.get("scratchpad", [])
        if scratchpad:
            with st.expander(f"🔍 Inspect Scratchpad & Tool Executions (Step {i+1})", expanded=False):
                for item in scratchpad:
                    if item.get("type") == "tool_call":
                        st.markdown(f"**🛠️ Tool Call:** `{item.get('name')}`")
                        st.json(item.get("args"))
                    elif item.get("type") == "tool_observation":
                        st.markdown(f"**👁️ Observation ({item.get('name')}):**")
                        st.code(str(item.get("content")), language="markdown")

    st.session_state["agentic_thinking_spinner"] = st.empty()


def clear_agentic_state():
    if "agentic_rag_helper" in st.session_state and st.session_state["agentic_rag_helper"]:
        st.session_state["agentic_rag_helper"].clear_documents()
    st.session_state["agentic_messages"] = []
    st.toast("Agentic RAG session and document knowledge base reset.")


# --- Main Application Layout ---
def page():
    # Session state initializations
    if "assistant" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    if "agent_messages" not in st.session_state:
        st.session_state["agent_messages"] = []

    if "agent_uploaded_files" not in st.session_state:
        st.session_state["agent_uploaded_files"] = []

    if "agentic_messages" not in st.session_state:
        st.session_state["agentic_messages"] = []

    if "agentic_rag_helper" not in st.session_state:
        st.session_state["agentic_rag_helper"] = AgenticRAGHelper() if AgenticRAGHelper is not None else None

    st.header("Local RAG & AI Agent Architectures")

    tab_rag, tab_agent, tab_agentic_rag = st.tabs(["RAG", "AGENT", "Agentic RAG"])

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

    with tab_agentic_rag:
        st.subheader("Complete Agent Architecture (LangGraph + Redis Checkpointer)")
        st.markdown(
            "> Explores **3. Complete Agent Architecture** from `Agent.md`: ReAct `AgentCore`, "
            "`4. Tools` (Knowledge retrieval, file reader, calculator), and `3. Short-term Memory` (Redis checkpointing)."
        )

        # Check Redis Status
        if st.session_state.get("agentic_rag_helper"):
            redis_status = st.session_state["agentic_rag_helper"].get_redis_status()
            if redis_status["active"]:
                st.success(f"🟢 **Redis Checkpointer Active**: {redis_status['url']} (Multi-turn state & scratchpads persisted)")
            else:
                st.warning(f"🟡 **In-Memory Checkpointer Fallback**: Redis not reachable at `{redis_status['url']}`. Start Docker Redis to enable persistent checkpointer storage.")

        # Visual Graph Topology Display (Matching langgraph_ollama style)
        if st.session_state.get("agentic_rag_helper") and render_agentic_rag_graph is not None:
            with st.expander("🗺️ Agentic RAG LangGraph Topology & Flow", expanded=False):
                render_agentic_rag_graph(
                    st.session_state["agentic_rag_helper"].core.graph,
                    caption="LangGraph ReAct Topology: __start__ ➔ agent (Ollama LLM) ⇄ tools (RAG / Files / Math) ➔ __end__"
                )

        col1, col2 = st.columns([3, 1])
        with col1:
            st.file_uploader(
                "Upload PDF Document for Agentic RAG",
                type=["pdf"],
                key="agentic_file_uploader",
                on_change=ingest_agentic_rag_files,
                label_visibility="collapsed",
                accept_multiple_files=True,
            )
        with col2:
            st.session_state["agentic_thread_id"] = st.text_input(
                "Session / Thread ID",
                value=st.session_state.get("agentic_thread_id", "session_1"),
                help="LangGraph thread_id used for Redis checkpoint persistence across turns",
            )

        if st.session_state.get("agentic_rag_helper"):
            ingested = st.session_state["agentic_rag_helper"].get_ingested_files()
            if ingested:
                st.caption(f"📚 **Ingested Knowledge Base Documents:** {', '.join(ingested)}")

        st.session_state["agentic_ingestion_spinner"] = st.empty()

        if AgenticRAGHelper is not None:
            display_agentic_rag_messages()
            st.text_input("Ask Agentic RAG", key="agentic_user_input", on_change=process_agentic_rag_input)
            st.button("Reset Agentic Session & Knowledge Base", on_click=clear_agentic_state)
        else:
            st.error("Agentic RAG module failed to load. Please check dependencies.")


if __name__ == "__main__":
    page()
