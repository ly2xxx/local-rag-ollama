"""LangGraph Visual Topology Renderer for Agentic RAG.

Extracts the Mermaid representation and PNG diagram of the compiled LangGraph state machine,
caching rendered diagrams on disk and falling back gracefully to Mermaid code blocks
if offline or if remote render services are unavailable.
"""

import hashlib
from io import BytesIO
from pathlib import Path
from typing import Optional, Any
import streamlit as st
from PIL import Image


def get_graph_mermaid(chain: Any, xray: bool = True) -> str:
    """Extracts Mermaid diagram string from a LangGraph Runnable/CompiledGraph."""
    try:
        graph = chain.get_graph(xray=xray)
        return graph.draw_mermaid()
    except Exception as e:
        return f"%% Error extracting mermaid: {e} %%\ngraph TD\n    __start__ --> agent\n    agent --> tools\n    tools --> agent\n    agent --> __end__"


def render_agentic_rag_graph(
    chain: Any,
    caption: str = "Agentic RAG LangGraph Topology (ReAct Loop)",
    height: int = 400,
) -> None:
    """Renders the LangGraph diagram in Streamlit with disk caching and fallback."""
    if chain is None:
        st.info("No active LangGraph chain provided.")
        return

    try:
        graph = chain.get_graph(xray=True)
        mermaid_src = graph.draw_mermaid()
    except Exception as e:
        st.warning(f"Could not extract graph topology: {e}")
        return

    cache_dir = Path(".cache/graph-png")
    cache_dir.mkdir(parents=True, exist_ok=True)
    png_path = cache_dir / (hashlib.sha256(mermaid_src.encode()).hexdigest()[:16] + ".png")

    image_loaded = False
    if not png_path.exists():
        try:
            png_bytes = graph.draw_mermaid_png()
            png_path.write_bytes(png_bytes)
            image_loaded = True
        except Exception:
            image_loaded = False
    else:
        image_loaded = True

    # If image is available, render PNG; otherwise render native markdown Mermaid
    if image_loaded and png_path.exists():
        try:
            image = Image.open(BytesIO(png_path.read_bytes()))
            aspect_ratio = image.width / max(image.height, 1)
            new_width = int(height * aspect_ratio)
            st.image(image.resize((new_width, height)), caption=caption)
        except Exception:
            image_loaded = False

    if not image_loaded:
        st.markdown(f"**{caption}**")
        st.markdown(f"```mermaid\n{mermaid_src}\n```")

    with st.expander("📝 View Mermaid Graph Source", expanded=False):
        st.code(mermaid_src, language="mermaid")
