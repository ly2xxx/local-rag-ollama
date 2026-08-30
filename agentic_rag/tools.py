"""4. Tools Component (Knowledge Retrieval, File Inspection & Computation).

Implements callable LangChain/LangGraph tools for the Agent:
1. Document Knowledge Base Retrieval (RAG vector search)
2. Local File Reading (PDF / text)
3. Mathematical Computation Tool
as described in Section 3 of Agent.md.

Knowledge bases are namespaced per session/tenant and persisted to disk, so that
concurrent users never share a vector store and ingested documents survive a
process restart.
"""

import ast
import hashlib
import logging
import math
import operator
import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Iterator, List, Optional

from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pypdf

from .config import (
    CHROMA_PERSIST_DIR,
    DEFAULT_NAMESPACE,
    EMBEDDING_MODEL,
    FILE_TOOL_ROOTS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared embedding model
# ---------------------------------------------------------------------------

_embeddings_lock = threading.Lock()
_embeddings_cache: Dict[str, HuggingFaceEmbeddings] = {}


def get_embeddings(model_name: Optional[str] = None) -> HuggingFaceEmbeddings:
    """Returns a process-wide shared embedding model.

    Loading a sentence-transformer costs hundreds of MB and several seconds, so
    it is shared across every namespace rather than instantiated per store.
    """
    name = model_name or EMBEDDING_MODEL
    with _embeddings_lock:
        if name not in _embeddings_cache:
            logger.info("Loading embedding model '%s'", name)
            _embeddings_cache[name] = HuggingFaceEmbeddings(model_name=name)
        return _embeddings_cache[name]


# ---------------------------------------------------------------------------
# Namespace scoping
# ---------------------------------------------------------------------------

# The active namespace travels with the execution context rather than being a
# tool argument, so the LLM can never select another tenant's knowledge base.
_active_namespace: ContextVar[str] = ContextVar(
    "agentic_rag_namespace", default=DEFAULT_NAMESPACE
)


def get_active_namespace() -> str:
    """Returns the namespace currently bound to this execution context."""
    return _active_namespace.get()


@contextmanager
def use_namespace(namespace: Optional[str]) -> Iterator[str]:
    """Binds a namespace for the duration of the block (contextvar scoped)."""
    resolved = (namespace or DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE
    token = _active_namespace.set(resolved)
    try:
        yield resolved
    finally:
        _active_namespace.reset(token)


_SLUG_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


def namespace_to_collection(namespace: str) -> str:
    """Maps an arbitrary namespace string to a valid Chroma collection name.

    Chroma requires alphanumeric start/end characters, so a hash suffix both
    guarantees validity and keeps distinct namespaces from colliding after
    slugification.
    """
    slug = _SLUG_PATTERN.sub("-", namespace.strip()).strip("-_").lower()[:32]
    digest = hashlib.sha1(namespace.encode("utf-8")).hexdigest()[:8]
    return f"kb-{slug}-{digest}" if slug else f"kb-{digest}"


# ---------------------------------------------------------------------------
# Document store
# ---------------------------------------------------------------------------


class DocumentRetrieverManager:
    """Manages a persisted Chroma collection for a single namespace."""

    def __init__(
        self,
        namespace: str = DEFAULT_NAMESPACE,
        embedding_model_name: Optional[str] = None,
        persist_directory: Optional[str] = None,
    ):
        self.namespace = namespace
        self.collection_name = namespace_to_collection(namespace)
        self.persist_directory = persist_directory or CHROMA_PERSIST_DIR
        self.embedding_model_name = embedding_model_name or EMBEDDING_MODEL
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self._vector_store: Optional[Chroma] = None
        self._store_lock = threading.Lock()

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        return get_embeddings(self.embedding_model_name)

    @property
    def vector_store(self) -> Chroma:
        """Lazily opens (or reopens) the persisted collection."""
        if self._vector_store is None:
            with self._store_lock:
                if self._vector_store is None:
                    os.makedirs(self.persist_directory, exist_ok=True)
                    self._vector_store = Chroma(
                        collection_name=self.collection_name,
                        embedding_function=self.embeddings,
                        persist_directory=self.persist_directory,
                    )
        return self._vector_store

    def as_retriever(self, top_k: int = 3) -> Any:
        return self.vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )

    def count(self) -> int:
        """Number of chunks currently stored in this namespace."""
        try:
            return self.vector_store._collection.count()
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("Could not count collection '%s': %s", self.collection_name, e)
            return 0

    @property
    def ingested_files(self) -> List[str]:
        """Document names, derived from the store so the list survives restarts."""
        try:
            records = self.vector_store.get(include=["metadatas"])
        except Exception as e:
            logger.warning("Could not list documents for '%s': %s", self.collection_name, e)
            return []

        names: List[str] = []
        for metadata in records.get("metadatas") or []:
            if not metadata:
                continue
            name = metadata.get("source_name")
            if not name and metadata.get("source"):
                name = os.path.basename(str(metadata["source"]))
            if name and name not in names:
                names.append(name)
        return sorted(names)

    @staticmethod
    def _chunk_id(source_name: str, chunk: Any) -> str:
        """Deterministic chunk id so re-ingesting the same file upserts in place."""
        page = chunk.metadata.get("page", "?")
        index = chunk.metadata.get("chunk_index", 0)
        payload = f"{source_name}|{page}|{index}|{chunk.page_content}"
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def ingest_pdf(self, file_path: str, display_name: Optional[str] = None) -> str:
        """Ingests a PDF file into this namespace's vector store."""
        if not os.path.exists(file_path):
            return f"Error: File not found at '{file_path}'"

        # Streamlit writes uploads to randomly-named temp files, so the caller
        # passes the real filename to keep listings and de-duplication stable.
        name = display_name or os.path.basename(file_path)

        try:
            docs = PyPDFLoader(file_path=file_path).load()
        except Exception as e:
            logger.error("Failed to load PDF '%s': %s", name, e, exc_info=True)
            return f"Error reading PDF '{name}': {e}"

        chunks = self.text_splitter.split_documents(docs)
        if not chunks:
            return f"No extractable text found in '{name}'."

        for index, chunk in enumerate(chunks):
            chunk.metadata["source_name"] = name
            chunk.metadata["chunk_index"] = index
        chunks = filter_complex_metadata(chunks)

        ids = [self._chunk_id(name, chunk) for chunk in chunks]
        try:
            self.vector_store.add_documents(chunks, ids=ids)
        except Exception as e:
            logger.error("Failed to index '%s': %s", name, e, exc_info=True)
            return f"Error indexing '{name}': {e}"

        return (
            f"Successfully ingested '{name}' ({len(chunks)} chunks) "
            f"into knowledge base '{self.namespace}'."
        )

    def query(self, query_text: str, top_k: int = 3) -> str:
        """Queries this namespace for the top relevant document chunks."""
        if self.count() == 0:
            return (
                "No documents have been ingested into the knowledge base yet. "
                "Please upload and ingest a document first."
            )

        results = self.vector_store.similarity_search(query_text, k=top_k)
        if not results:
            return f"No matching passages found in the knowledge base for query: '{query_text}'."

        formatted = []
        for i, doc in enumerate(results, start=1):
            source = doc.metadata.get("source_name") or os.path.basename(
                str(doc.metadata.get("source", "Unknown Source"))
            )
            page = doc.metadata.get("page", "N/A")
            formatted.append(
                f"--- [Document Match #{i}] Source: {source} (Page {page}) ---\n"
                f"{doc.page_content.strip()}"
            )

        return "\n\n".join(formatted)

    def clear(self):
        """Deletes this namespace's collection from disk."""
        try:
            self.vector_store.delete_collection()
        except Exception as e:
            logger.warning("Could not delete collection '%s': %s", self.collection_name, e)
        finally:
            with self._store_lock:
                self._vector_store = None


class DocumentStoreRegistry:
    """Maps namespaces to their own `DocumentRetrieverManager` instances."""

    def __init__(self):
        self._managers: Dict[str, DocumentRetrieverManager] = {}
        self._lock = threading.Lock()

    def get(self, namespace: Optional[str] = None) -> DocumentRetrieverManager:
        resolved = (namespace or get_active_namespace() or DEFAULT_NAMESPACE).strip()
        resolved = resolved or DEFAULT_NAMESPACE
        with self._lock:
            if resolved not in self._managers:
                self._managers[resolved] = DocumentRetrieverManager(namespace=resolved)
            return self._managers[resolved]

    def drop(self, namespace: str):
        """Clears and forgets a namespace entirely."""
        with self._lock:
            manager = self._managers.pop(namespace, None)
        if manager is not None:
            manager.clear()

    def namespaces(self) -> List[str]:
        with self._lock:
            return sorted(self._managers)


doc_registry = DocumentStoreRegistry()


def get_doc_manager(namespace: Optional[str] = None) -> DocumentRetrieverManager:
    """Returns the document store for a namespace (defaults to the active one)."""
    return doc_registry.get(namespace)


# ---------------------------------------------------------------------------
# Safe arithmetic evaluation
# ---------------------------------------------------------------------------

MAX_EXPRESSION_LENGTH = 500
MAX_EXPRESSION_NODES = 200
MAX_EXPONENT = 1000

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

_ALLOWED_FUNCTIONS = {
    "sum": sum,
    "min": min,
    "max": max,
    "abs": abs,
    "round": round,
    "len": len,
    "pow": pow,
    "int": int,
    "float": float,
}

_ALLOWED_MODULES = {"math": math}


class UnsafeExpressionError(ValueError):
    """Raised when an expression falls outside the calculator whitelist."""


def _resolve_attribute(node: ast.Attribute) -> Any:
    """Resolves `math.<name>`; every other attribute access is rejected."""
    if not isinstance(node.value, ast.Name) or node.value.id not in _ALLOWED_MODULES:
        raise UnsafeExpressionError("Only attributes of the 'math' module are allowed.")
    if node.attr.startswith("_"):
        raise UnsafeExpressionError("Access to private attributes is not allowed.")
    module = _ALLOWED_MODULES[node.value.id]
    if not hasattr(module, node.attr):
        raise UnsafeExpressionError(f"'math.{node.attr}' does not exist.")
    return getattr(module, node.attr)


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool)):
            return node.value
        if isinstance(node.value, str) and len(node.value) <= 256:
            return node.value
        raise UnsafeExpressionError("Unsupported constant in expression.")

    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(
                f"Operator '{type(node.op).__name__}' is not allowed."
            )
        left, right = _eval_node(node.left), _eval_node(node.right)
        # Guard against `9**9**9`-style resource exhaustion.
        if isinstance(node.op, ast.Pow) and isinstance(right, (int, float)):
            if abs(right) > MAX_EXPONENT:
                raise UnsafeExpressionError(
                    f"Exponent exceeds the maximum of {MAX_EXPONENT}."
                )
        return op(left, right)

    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise UnsafeExpressionError(
                f"Unary operator '{type(node.op).__name__}' is not allowed."
            )
        return op(_eval_node(node.operand))

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left)
        for op_node, comparator in zip(node.ops, node.comparators):
            op = _COMPARE_OPS.get(type(op_node))
            if op is None:
                raise UnsafeExpressionError(
                    f"Comparison '{type(op_node).__name__}' is not allowed."
                )
            right = _eval_node(comparator)
            if not op(left, right):
                return False
            left = right
        return True

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(element) for element in node.elts]

    if isinstance(node, ast.Call):
        if node.keywords:
            raise UnsafeExpressionError("Keyword arguments are not allowed.")
        if isinstance(node.func, ast.Name):
            func = _ALLOWED_FUNCTIONS.get(node.func.id)
            if func is None:
                raise UnsafeExpressionError(f"Function '{node.func.id}' is not allowed.")
        elif isinstance(node.func, ast.Attribute):
            func = _resolve_attribute(node.func)
            if not callable(func):
                raise UnsafeExpressionError("Attempted to call a non-callable value.")
        else:
            raise UnsafeExpressionError("Unsupported call target.")
        return func(*[_eval_node(arg) for arg in node.args])

    if isinstance(node, ast.Attribute):
        value = _resolve_attribute(node)
        if callable(value):
            raise UnsafeExpressionError(
                "Functions must be called, not referenced (e.g. 'math.sqrt(9)')."
            )
        return value

    raise UnsafeExpressionError(
        f"Expression element '{type(node).__name__}' is not allowed."
    )


def safe_eval_expression(expression: str) -> Any:
    """Evaluates an arithmetic expression against a strict AST whitelist.

    Unlike `eval` with an emptied `__builtins__`, this never resolves arbitrary
    names or attributes, so the usual sandbox escapes (walking `__class__` /
    `__subclasses__` off a literal) have no reachable node type.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise UnsafeExpressionError(
            f"Expression exceeds {MAX_EXPRESSION_LENGTH} characters."
        )

    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise UnsafeExpressionError(f"Invalid syntax: {e.msg}") from e

    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise UnsafeExpressionError("Expression is too complex to evaluate.")

    return _eval_node(tree)


# ---------------------------------------------------------------------------
# File access scoping
# ---------------------------------------------------------------------------


def _allowed_file_roots() -> List[str]:
    roots = [part for part in FILE_TOOL_ROOTS.split(os.pathsep) if part.strip()]
    return [os.path.realpath(root) for root in roots]


def _is_readable_path(path: str) -> bool:
    """Confines the file tool to configured roots.

    Documents fed to the agent are untrusted input; without this an injected
    instruction could make the model read arbitrary files off the host.
    """
    roots = _allowed_file_roots()
    if not roots:
        return True

    real_path = os.path.realpath(path)
    for root in roots:
        try:
            if os.path.commonpath([real_path, root]) == root:
                return True
        except ValueError:
            # Different drives on Windows have no common path.
            continue
    return False


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def query_document_knowledge_base(query: str) -> str:
    """Search and retrieve relevant context from the ingested PDF/document knowledge base.

    Args:
        query: The semantic search question or keywords to look up in the documents.

    Returns:
        Relevant document snippets and citation details.
    """
    return doc_registry.get().query(query)


@tool
def read_local_file(file_path: str) -> str:
    """Reads a local PDF, text, or markdown file and returns its raw text contents.

    Args:
        file_path: The local filesystem path to the file.

    Returns:
        Extracted text contents of the file.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path '{file_path}'"

    if not _is_readable_path(file_path):
        return (
            f"Error: Access denied. '{file_path}' is outside the permitted "
            f"directories: {', '.join(_allowed_file_roots())}"
        )

    try:
        if file_path.lower().endswith(".pdf"):
            reader = pypdf.PdfReader(file_path)
            pages = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text()
                if txt and txt.strip():
                    pages.append(f"--- Page {idx + 1} ---\n{txt.strip()}")
            return "\n\n".join(pages) if pages else f"PDF '{file_path}' has no readable text."
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        return f"Error reading file '{file_path}': {str(e)}"


@tool
def calculate_expression(expression: str) -> str:
    """Evaluates a mathematical expression or calculation safely.

    Args:
        expression: The mathematical expression to compute (e.g. 'sum([80, 90, 99, 70])/4' or 'math.sqrt(144)').

    Returns:
        The calculated result as a string.
    """
    try:
        result = safe_eval_expression(expression)
    except UnsafeExpressionError as e:
        return f"Rejected expression '{expression}': {e}"
    except (ArithmeticError, TypeError, ValueError) as e:
        return f"Error calculating expression '{expression}': {e}"
    return f"Calculation Result: {result}"


def get_default_tools() -> List[Any]:
    """Returns the list of default callable tools for the agent."""
    return [query_document_knowledge_base, read_local_file, calculate_expression]
