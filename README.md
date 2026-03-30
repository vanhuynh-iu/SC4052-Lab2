# Fast-GraphRAG Demo

SC4052 Lab 2 — A Streamlit application for building and querying knowledge graphs using [fast-graphrag](https://github.com/circlemind-ai/fast-graphrag) with PageRank-based entity retrieval.

## Overview

This app lets you ingest text documents into a knowledge graph, then query them using multi-hop reasoning. Entities extracted from your documents are ranked by PageRank score, and the most relevant are passed to an LLM to answer complex questions that require connecting information across multiple sources.

## Features

- **Knowledge Base tab** — Paste text, upload `.txt`/`.md`/`.pdf` files, or load a built-in physics history sample to populate the graph
- **Query tab** — Ask multi-hop questions with adjustable top-k entity retrieval (1–10); each answer shows the retrieved entities and their PageRank scores
- **Graph tab** — Interactive force-directed visualization of the knowledge graph with nodes sized and colored by entity type and PageRank score

## Requirements

- Python 3.12+
- An OpenAI API key (used by fast-graphrag for entity extraction and answering)

## Setup

1. Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

2. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

3. Run the app:

```bash
streamlit run app.py
```

## Project Structure

```
app.py          # Streamlit UI (three tabs: Knowledge Base, Query, Graph)
backend.py      # GraphRAGService wrapper around fast-graphrag
requirements.txt
.env            # API key (not committed)
knowledge_graph/  # Persisted graph data (auto-created on first insert)
```

## How It Works

1. **Ingestion** — `GraphRAGService.insert()` calls `fast_graphrag.GraphRAG.insert()`, which extracts entities and relationships from your text and stores them in an igraph-backed knowledge graph on disk.
2. **Querying** — `GraphRAGService.query()` runs PageRank over the graph to surface the top-k most relevant entities for a given question, then passes those entities and their relationships as context to the LLM.
3. **Visualization** — `get_graph_data()` loads the persisted graph, computes global PageRank scores, and returns nodes/edges for rendering with Plotly and NetworkX.

## Example Queries (Physics History Sample)

- *What discoveries by Marie Curie led to advances in medical imaging?*
- *How is Einstein's work connected to the Manhattan Project through other scientists?*
- *Trace the path from radioactivity discovery to nuclear reactors.*
