"""
RAG (Retrieval-Augmented Generation) MCP Server - Provides document retrieval tools.

This MCP server exposes RAG tools via HTTP transport (streamable-http).
It simulates a document database with search and retrieval capabilities.

Usage:
    python rag_server.py

    The server will run on http://localhost:8006/mcp

Configuration in agent YAML:
    mcp_servers:
      - name: rag
        description: Document retrieval and search
        transport: streamable_http
        url: http://localhost:8006/mcp
"""

from typing import List, Dict, Any
import asyncio
import uvicorn
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP("RAG")

# Dummy document database
DOCUMENT_DATABASE = [
    {
        "id": "doc_1",
        "title": "Introduction to LangChain",
        "content": "LangChain is a framework for developing applications powered by language models. It provides tools for building context-aware applications that can reason and interact with external data sources.",
        "category": "ai",
        "tags": ["langchain", "llm", "framework"]
    },
    {
        "id": "doc_2",
        "title": "Understanding Vector Databases",
        "content": "Vector databases store embeddings and enable similarity search. They are essential for RAG applications, allowing fast retrieval of relevant documents based on semantic similarity.",
        "category": "database",
        "tags": ["vector-db", "embeddings", "rag"]
    },
    {
        "id": "doc_3",
        "title": "Prompt Engineering Best Practices",
        "content": "Effective prompts are clear, specific, and provide context. Use examples when possible, break complex tasks into steps, and iterate based on results.",
        "category": "ai",
        "tags": ["prompts", "llm", "best-practices"]
    },
    {
        "id": "doc_4",
        "title": "Python AsyncIO Guide",
        "content": "AsyncIO enables concurrent programming in Python. It's particularly useful for I/O-bound operations, allowing multiple tasks to run concurrently without threading overhead.",
        "category": "programming",
        "tags": ["python", "async", "concurrency"]
    },
    {
        "id": "doc_5",
        "title": "Building Agentic Systems",
        "content": "AI agents can use tools, maintain memory, and make decisions. Modern agents combine LLMs with function calling, enabling them to interact with external systems and perform complex tasks.",
        "category": "ai",
        "tags": ["agents", "llm", "tools"]
    },
    {
        "id": "doc_6",
        "title": "FastAPI Framework Overview",
        "content": "FastAPI is a modern Python web framework. It provides automatic API documentation, data validation with Pydantic, and excellent performance through async support.",
        "category": "programming",
        "tags": ["fastapi", "python", "web"]
    },
    {
        "id": "doc_7",
        "title": "Semantic Search Techniques",
        "content": "Semantic search finds results based on meaning rather than keywords. It uses embeddings to represent text in vector space, enabling similarity-based retrieval.",
        "category": "ai",
        "tags": ["search", "embeddings", "nlp"]
    },
    {
        "id": "doc_8",
        "title": "Streamlit for Data Apps",
        "content": "Streamlit enables rapid development of data applications and dashboards. It provides simple APIs for creating interactive UIs without frontend complexity.",
        "category": "programming",
        "tags": ["streamlit", "python", "ui"]
    }
]


def _simple_search(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Simple keyword-based search (mock implementation).
    In a real RAG system, this would use embedding similarity.
    """
    query_lower = query.lower()
    results = []

    for doc in documents:
        # Check if query terms appear in title, content, or tags
        text = f"{doc['title']} {doc['content']} {' '.join(doc['tags'])}".lower()
        if query_lower in text or any(term in text for term in query_lower.split()):
            # Calculate simple relevance score (mock)
            score = sum(1 for term in query_lower.split() if term in text)
            results.append({**doc, "relevance_score": score})

    # Sort by relevance score
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    return results


@mcp.tool()
async def search_documents(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Search for documents relevant to a query.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default: 5)

    Returns:
        List of relevant documents with metadata and relevance scores

    Examples:
        search_documents("langchain framework")
        search_documents("vector database", max_results=3)
    """
    # Simulate async processing
    await asyncio.sleep(0.1)

    results = _simple_search(query, DOCUMENT_DATABASE)
    return results[:max_results]


@mcp.tool()
async def retrieve_context(query: str, max_chunks: int = 3) -> str:
    """
    Retrieve context chunks for a query, formatted for LLM consumption.

    Args:
        query: The search query
        max_chunks: Maximum number of context chunks to retrieve (default: 3)

    Returns:
        Formatted context string ready to be added to a prompt

    Examples:
        retrieve_context("what is langchain")
        retrieve_context("async programming in python", max_chunks=2)
    """
    # Simulate async processing
    await asyncio.sleep(0.1)

    results = _simple_search(query, DOCUMENT_DATABASE)
    top_results = results[:max_chunks]

    if not top_results:
        return "No relevant context found for the query."

    # Format context for LLM
    context_parts = ["Retrieved context:"]
    for i, doc in enumerate(top_results, 1):
        context_parts.append(f"\n[{i}] {doc['title']}")
        context_parts.append(f"{doc['content']}")

    return "\n".join(context_parts)


@mcp.tool()
async def get_relevant_chunks(
    query: str,
    category: str = None,
    max_results: int = 5
) -> List[str]:
    """
    Get relevant text chunks matching a query and optional category filter.

    Args:
        query: The search query
        category: Optional category filter (ai, database, programming)
        max_results: Maximum number of chunks to return (default: 5)

    Returns:
        List of relevant text chunks

    Examples:
        get_relevant_chunks("embeddings")
        get_relevant_chunks("python", category="programming")
    """
    # Simulate async processing
    await asyncio.sleep(0.1)

    # Filter by category if specified
    documents = DOCUMENT_DATABASE
    if category:
        documents = [doc for doc in documents if doc.get("category") == category.lower()]

    results = _simple_search(query, documents)
    return [doc["content"] for doc in results[:max_results]]


@mcp.tool()
async def list_document_categories() -> List[str]:
    """
    List all available document categories in the database.

    Returns:
        List of unique categories

    Examples:
        list_document_categories() -> ["ai", "database", "programming"]
    """
    categories = set(doc.get("category", "uncategorized") for doc in DOCUMENT_DATABASE)
    return sorted(categories)


@mcp.tool()
async def get_document_by_id(doc_id: str) -> Dict[str, Any]:
    """
    Retrieve a specific document by its ID.

    Args:
        doc_id: The document ID

    Returns:
        Document data with full content and metadata

    Raises:
        ValueError: If document ID not found

    Examples:
        get_document_by_id("doc_1")
    """
    for doc in DOCUMENT_DATABASE:
        if doc["id"] == doc_id:
            return doc

    raise ValueError(f"Document with ID '{doc_id}' not found")


@mcp.tool()
async def get_documents_by_tag(tag: str) -> List[Dict[str, Any]]:
    """
    Retrieve all documents with a specific tag.

    Args:
        tag: The tag to filter by

    Returns:
        List of documents matching the tag

    Examples:
        get_documents_by_tag("langchain")
        get_documents_by_tag("python")
    """
    tag_lower = tag.lower()
    return [
        doc for doc in DOCUMENT_DATABASE
        if tag_lower in [t.lower() for t in doc.get("tags", [])]
    ]


@mcp.tool()
async def summarize_documents(query: str, max_docs: int = 3) -> str:
    """
    Search for documents and provide a summary of their contents.

    Args:
        query: The search query
        max_docs: Maximum number of documents to summarize (default: 3)

    Returns:
        A summary combining information from relevant documents

    Examples:
        summarize_documents("langchain and agents")
        summarize_documents("vector databases", max_docs=2)
    """
    # Simulate async processing
    await asyncio.sleep(0.15)

    results = _simple_search(query, DOCUMENT_DATABASE)
    top_results = results[:max_docs]

    if not top_results:
        return "No relevant documents found for the query."

    # Create summary
    summary_parts = [f"Summary of {len(top_results)} document(s) related to '{query}':"]

    for doc in top_results:
        summary_parts.append(f"\n• {doc['title']}: {doc['content'][:150]}...")

    return "\n".join(summary_parts)


if __name__ == "__main__":
    # Run the MCP server using streamable-http transport
    # This starts an independent HTTP server on port 8006
    print("Starting RAG MCP Server on http://localhost:8006/mcp")
    print(f"Document database contains {len(DOCUMENT_DATABASE)} documents")
    print("Available tools: search_documents, retrieve_context, get_relevant_chunks, and more")

    # Use uvicorn to run the streamable-http app on the specified port
    uvicorn.run(mcp.streamable_http_app, host="0.0.0.0", port=8006)
