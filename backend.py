"""
GraphRAG Backend — corrected for fast-graphrag 0.0.4.

Key facts about fast-graphrag 0.0.4:
  - grag.insert() and grag.query() are SYNCHRONOUS (they wrap async internally)
  - Keyword is `params=` (not `param=`)
  - State manager lives at grag.state_manager (not _state_manager)
  - Graph storage at grag.state_manager.graph_storage
  - graph_storage internal methods (node_count, get_all_edges, etc.) ARE coroutines
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a coroutine safely from a sync context
# ─────────────────────────────────────────────────────────────────────────────

def _run_async(coro):
    """Run a coroutine in a fresh thread so we never fight an existing loop."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


# ─────────────────────────────────────────────────────────────────────────────
# Service
# ─────────────────────────────────────────────────────────────────────────────

class GraphRAGService:
    """
    Thin wrapper around fast_graphrag.GraphRAG (v0.0.4).

    insert() and query() are called synchronously.
    Graph storage introspection uses _run_async() for the async storage methods.
    """

    _ENTITY_TYPES = [
        "Person", "Organization", "Location", "Event",
        "Discovery", "Technology", "Concept", "Disease",
        "Chemical Element", "Medical Procedure",
    ]

    def __init__(self, working_dir: str = "./knowledge_graph"):
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._grag = None

    def _instance(self):
        if self._grag is None:
            from fast_graphrag import GraphRAG  # type: ignore
            self._grag = GraphRAG(
                working_dir=str(self.working_dir),
                domain=(
                    "A general-purpose knowledge base supporting multi-hop reasoning "
                    "across scientific, historical, and technological domains."
                ),
                example_queries=(
                    "What discoveries by Marie Curie led to advances in medical imaging?\n"
                    "How did scientific breakthroughs in physics influence modern technology?\n"
                    "Which organisations contributed to early radioactivity research?"
                ),
                entity_types=self._ENTITY_TYPES,
            )
        return self._grag

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def insert(self, text: str, source_name: str = "document") -> dict:
        """Insert a document. grag.insert() is synchronous in 0.0.4."""
        grag = self._instance()
        try:
            grag.insert(text, metadata=[{"source": source_name}])
            return {"success": True, "message": f'"{source_name}" ingested successfully.'}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, question: str, top_k: int = 10) -> dict:
        """
        Query the graph using fast-graphrag.
        Returns the response with entities ranked by PageRank.
        """
        from fast_graphrag import QueryParam  # type: ignore

        grag = self._instance()

        # Query the graph
        try:
            response = grag.query(
                question,
                params=QueryParam(
                    with_references=True,
                    entities_max_tokens=4000,
                    relationships_max_tokens=3000,
                    chunks_max_tokens=9000,
                ),
            )
        except Exception as exc:
            return {"answer": f"Query failed: {exc}", "entities": [], "relationships": []}

        # Extract entities with PageRank scores
        entities: list[dict] = []
        if response.context and response.context.entities:
            for entity, score in response.context.entities[:top_k]:
                entities.append({
                    "name":           entity.name,
                    "type":           getattr(entity, "type", "Unknown"),
                    "description":    getattr(entity, "description", ""),
                    "pagerank_score": float(score),
                })

        # Extract relationships
        relationships: list[dict] = []
        if response.context and hasattr(response.context, 'relationships') and response.context.relationships:
            for rel, score in response.context.relationships[:top_k * 2]:
                relationships.append({
                    "source":      rel.source,
                    "target":      rel.target,
                    "description": getattr(rel, "description", ""),
                    "score":       float(score),
                })

        # Use fast-graphrag's response directly (it already has citations)
        answer = response.response if response.response else "No relevant information found in the knowledge graph."

        return {
            "answer":    answer,
            "entities":  entities,
            "relationships": relationships,
        }

    # ------------------------------------------------------------------
    # Graph data (for Graph Explorer)
    # ------------------------------------------------------------------

    def get_graph_data(self) -> dict:
        """
        Extract nodes + edges from the igraph instance for visualisation.
        Global PageRank is computed with uniform seed (no query context).
        graph_storage internal methods are async, so we use _run_async().
        """
        grag = self._instance()
        try:
            gs = grag.state_manager.graph_storage

            # Load graph from disk (query_start loads the persisted data)
            async def load_and_get():
                await gs.query_start()
                g = gs._graph
                if g is None:
                    return None, None
                scores = await gs.score_nodes(None)
                await gs.query_done()
                return g, scores

            g, scores_raw = _run_async(load_and_get())

            if g is None:
                return {"nodes": [], "edges": [], "error": "No graph data found"}

            nodes: list[dict] = []
            for v in g.vs:
                name  = v["name"]        if "name"        in v.attributes() else str(v.index)
                etype = v["type"]        if "type"        in v.attributes() else "Unknown"
                desc  = v["description"] if "description" in v.attributes() else ""
                try:
                    score = float(scores_raw[v.index, 0]) if scores_raw is not None else 0.0
                except Exception:
                    score = 0.0
                nodes.append({"id": name, "type": etype, "description": desc, "score": score})

            edges: list[dict] = []
            for e in g.es:
                src = g.vs[e.source]["name"] if "name" in g.vs[e.source].attributes() else str(e.source)
                tgt = g.vs[e.target]["name"] if "name" in g.vs[e.target].attributes() else str(e.target)
                desc = e["description"] if "description" in e.attributes() else ""
                edges.append({"source": src, "target": tgt, "description": desc})

            return {"nodes": nodes, "edges": edges}

        except Exception as exc:
            return {"nodes": [], "edges": [], "error": str(exc)}

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict:
        try:
            gs = self._instance().state_manager.graph_storage

            async def get_counts():
                await gs.query_start()
                g = gs._graph
                if g is None:
                    return 0, 0
                counts = g.vcount(), g.ecount()
                await gs.query_done()
                return counts

            nodes, edges = _run_async(get_counts())
            return {"nodes": nodes, "edges": edges}
        except Exception:
            return {"nodes": 0, "edges": 0}

    # ------------------------------------------------------------------
    # Simple Chat (no retrieval)
    # ------------------------------------------------------------------

    def simple_chat(self, message: str) -> str:
        """Simple chat without graph retrieval - just uses the LLM."""
        try:
            from openai import OpenAI
            client = OpenAI()

            system_prompt = """You are a helpful AI assistant. You help users with general questions.

Note: This chat does not have access to any uploaded documents. If the user asks about specific documents or data they uploaded, politely let them know they need to upload documents first in the Documents tab to query them.

Be concise, friendly, and helpful."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
