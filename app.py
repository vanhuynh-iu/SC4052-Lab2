"""
Fast-GraphRAG Demo — PageRank-based Knowledge Graph Retrieval
SC4052 Lab 2: Multi-hop query answering with top-k entity ranking
"""

import os
import re
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv(Path(__file__).parent / ".env")

# Load API key from Streamlit secrets (for cloud) or .env (for local)
if "OPENAI_API_KEY" not in os.environ:
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Page config - at top to avoid blinking
st.set_page_config(
    page_title="Fast-GraphRAG Demo",
    page_icon=":material/hub:",
    layout="wide",
)

# Minimal CSS
st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.5rem !important;}
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [("service", None), ("messages", []), ("documents", []), ("graph_data", None), ("stats", {"nodes": 0, "edges": 0}), ("session_id", None)]:
    if key not in st.session_state:
        st.session_state[key] = default

# Generate unique session ID and clear old data on new session
import uuid
import shutil

WORKING_DIR = "./knowledge_graph"

# For development: don't clear data if it exists and has content
if st.session_state.session_id is None:
    st.session_state.session_id = str(uuid.uuid4())
    # Only clear if folder is empty or doesn't have graph data
    graph_file = os.path.join(WORKING_DIR, "graph_igraph_data.pklz")
    if not os.path.exists(graph_file):
        if os.path.exists(WORKING_DIR):
            shutil.rmtree(WORKING_DIR)
        os.makedirs(WORKING_DIR, exist_ok=True)

# Backend
def get_service():
    if st.session_state.service is None:
        from backend import GraphRAGService
        st.session_state.service = GraphRAGService(working_dir=WORKING_DIR)
        update_stats()
    return st.session_state.service

def update_stats():
    try:
        svc = st.session_state.service
        if svc:
            st.session_state.stats = svc.stats
    except:
        st.session_state.stats = {"nodes": 0, "edges": 0}

# Initialize service and always update stats
get_service()
update_stats()

def extract_citations(text):
    return [int(m) for m in re.findall(r'\[(\d+)\]', text)]

def strip_citations(text):
    """Remove citation markers like [1], [2], [3] from text."""
    return re.sub(r'\s*\[\d+\]', '', text)

# Header
api_ok = bool(os.environ.get("OPENAI_API_KEY"))
stats = st.session_state.stats

st.title("Fast-GraphRAG")
st.caption("PageRank-based retrieval for multi-hop knowledge graph queries · [circlemind-ai/fast-graphrag](https://github.com/circlemind-ai/fast-graphrag)")

# Tabs with icons
tab1, tab2, tab3 = st.tabs([
    ":material/description: Knowledge Base",
    ":material/chat: Query",
    ":material/hub: Graph"
])

# ═══════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns([2, 1], gap="large")

    with col_l:
        st.subheader("Add to knowledge graph")

        input_method = st.segmented_control(
            "Method",
            ["Paste text", "Upload file"],
            default="Paste text",
            label_visibility="collapsed"
        )

        text_content, doc_name = "", "my_document"

        if input_method == "Paste text":
            text_content = st.text_area(
                "Document text",
                height=180,
                placeholder="Paste text to extract entities and relationships...",
                label_visibility="collapsed"
            )
            doc_name = st.text_input("Source name", value="my_document")
        else:
            uploaded = st.file_uploader(
                "Choose a file",
                type=["txt", "md", "pdf"],
                label_visibility="collapsed"
            )
            if uploaded:
                doc_name = Path(uploaded.name).stem
                if uploaded.type == "application/pdf":
                    try:
                        import PyPDF2
                        text_content = "\n".join(p.extract_text() or "" for p in PyPDF2.PdfReader(uploaded).pages)
                    except:
                        st.error("Install pypdf2", icon=":material/error:")
                else:
                    text_content = uploaded.read().decode("utf-8", errors="replace")
                st.caption(f"{len(text_content):,} characters loaded")

        if st.button("Build knowledge graph", type="primary", icon=":material/upload:"):
            if not api_ok:
                st.error("Add OPENAI_API_KEY to .env file", icon=":material/key_off:")
            elif not text_content.strip():
                st.warning("Please add some text first", icon=":material/warning:")
            else:
                with st.spinner("Extracting entities and relationships..."):
                    result = get_service().insert(text_content, source_name=doc_name)
                    if result["success"]:
                        st.session_state.documents.append({"name": doc_name, "chars": len(text_content)})
                        update_stats()
                        st.toast(f"'{doc_name}' added to graph", icon=":material/check:")
                        st.balloons()
                    else:
                        st.error(result["message"], icon=":material/error:")

    with col_r:
        st.subheader("Example: Physics history")
        st.caption("Load sample data to explore multi-hop queries like: *'What discoveries by Marie Curie led to advances in medical imaging?'*")

        if st.button("Load sample data", icon=":material/science:", use_container_width=True):
            sample = """
Marie Curie (1867-1934) was a Polish-French physicist and chemist who pioneered research on radioactivity. Born Maria Sklodowska in Warsaw, Poland, she moved to Paris in 1891 to study at the Sorbonne. She married Pierre Curie in 1895, and together they discovered two new elements: polonium (named after Poland) and radium in 1898.

Marie Curie won two Nobel Prizes: the Nobel Prize in Physics (1903) shared with Pierre Curie and Henri Becquerel for their work on radioactivity, and the Nobel Prize in Chemistry (1911) for discovering radium and polonium. She was the first woman to win a Nobel Prize and remains the only person to win Nobel Prizes in two different sciences.

The Curie Institute in Paris, founded in 1920, continues research in cancer treatment using radiation therapy. Marie's daughter Irène Joliot-Curie also won a Nobel Prize in Chemistry (1935) with her husband Frédéric Joliot for discovering artificial radioactivity.

Ernest Rutherford, a New Zealand-born British physicist at Cambridge University, built on Curie's work to develop the nuclear model of the atom in 1911. His famous gold foil experiment at the Cavendish Laboratory proved that atoms have a dense nucleus. Rutherford won the Nobel Prize in Chemistry (1908) and mentored many future Nobel laureates including Niels Bohr and James Chadwick.

Henri Becquerel discovered radioactivity in 1896 while studying phosphorescent materials at the French Academy of Sciences. He found that uranium salts emitted rays that could fog photographic plates without exposure to sunlight. This discovery led directly to the Curies' research.

Albert Einstein, working at the Swiss Patent Office in Bern, published his theory of special relativity in 1905. His famous equation E=mc² showed the equivalence of mass and energy, which later explained the enormous energy released in nuclear reactions. Einstein won the Nobel Prize in Physics (1921) for his explanation of the photoelectric effect.

Niels Bohr developed the Bohr model of the atom at the University of Copenhagen in 1913, incorporating quantum theory into atomic structure. He won the Nobel Prize in Physics (1922) and founded the Copenhagen Interpretation of quantum mechanics. His institute became a major center for theoretical physics.

Enrico Fermi, an Italian physicist who later moved to the University of Chicago, created the first nuclear reactor (Chicago Pile-1) in 1942. He won the Nobel Prize in Physics (1938) for his work on induced radioactivity. The element fermium is named after him.

Lise Meitner, an Austrian-Swedish physicist, worked with Otto Hahn at the Kaiser Wilhelm Institute in Berlin. She provided the first theoretical explanation of nuclear fission in 1939, though she was overlooked for the Nobel Prize. Element 109 (meitnerium) honors her contributions.

The Manhattan Project, led by J. Robert Oppenheimer at Los Alamos National Laboratory, developed the first nuclear weapons during World War II. The project employed many scientists including Fermi, Bohr, and Richard Feynman. The Trinity test in New Mexico on July 16, 1945, was the first detonation of a nuclear device.

CERN (European Organization for Nuclear Research), located near Geneva, Switzerland, operates the Large Hadron Collider, the world's largest particle accelerator. CERN discovered the Higgs boson in 2012, confirming the existence of the Higgs field theorized by Peter Higgs and others.
"""
            if api_ok:
                with st.spinner("Loading sample (this may take a minute)..."):
                    result = get_service().insert(sample, source_name="physics_history")
                    if result["success"]:
                        st.session_state.documents.append({"name": "physics_history", "chars": len(sample)})
                        update_stats()
                        st.toast("Sample loaded! Go to Chat tab", icon=":material/check:")
            else:
                st.error("Add API key first", icon=":material/key_off:")

        if st.session_state.documents:
            st.subheader("Processed documents")
            for d in reversed(st.session_state.documents[-5:]):
                st.markdown(f":material/description: **{d['name']}** ({d['chars']:,} chars)")
        else:
            st.caption("No documents processed yet")


# ═══════════════════════════════════════════════════════════════════════════
# QUERY TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    has_docs = stats["nodes"] > 0

    # Suggestions mapping - multi-hop query examples
    SUGGESTIONS = {
        ":blue[:material/route:] Marie Curie → Medical imaging": "What discoveries by Marie Curie led to advances in medical imaging?",
        ":green[:material/person:] Key scientists": "Who are the key scientists and how did their work influence each other?",
        ":orange[:material/link:] Multi-hop connections": "How is Einstein's work connected to the Manhattan Project through other scientists?",
        ":violet[:material/hub:] Knowledge paths": "Trace the path from radioactivity discovery to nuclear reactors.",
    }

    # Check states
    user_just_asked = "initial_q" in st.session_state and st.session_state.initial_q
    user_clicked_pill = "selected_pill" in st.session_state and st.session_state.selected_pill
    first_interaction = user_just_asked or user_clicked_pill
    has_history = len(st.session_state.messages) > 0

    # INITIAL STATE - before any messages
    if not first_interaction and not has_history:
        # Centered welcome
        st.markdown("")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if has_docs:
                st.markdown(f"### :material/hub: Multi-hop Query")
                st.caption(f"PageRank retrieval over {stats['nodes']} entities · top-k ranking")
            else:
                st.markdown("### :material/chat: Query")
                st.caption("Load documents first to enable graph-based retrieval")

            st.markdown("")

            # Chat input
            st.chat_input("Ask a multi-hop question...", key="initial_q")

            # Suggestion pills
            if has_docs:
                st.pills(
                    "Example queries",
                    options=list(SUGGESTIONS.keys()),
                    label_visibility="collapsed",
                    key="selected_pill",
                )

        st.stop()

    # CONVERSATION STATE - after first message
    # Header row with restart button
    col1, col2 = st.columns([4, 1])
    with col1:
        if has_docs:
            st.caption(f":material/hub: PageRank retrieval · {stats['nodes']} entities · top-k ranking")
        else:
            st.caption(":material/chat: Direct LLM mode (no graph)")
    with col2:
        def restart_chat():
            st.session_state.messages = []
            st.session_state.initial_q = None
            st.session_state.selected_pill = None

        st.button("Clear", icon=":material/refresh:", on_click=restart_chat)

    # Get user message from initial input or pill
    user_message = None
    if user_just_asked:
        user_message = st.session_state.initial_q
        st.session_state.initial_q = None
    elif user_clicked_pill:
        user_message = SUGGESTIONS.get(st.session_state.selected_pill)
        st.session_state.selected_pill = None

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Show top-k retrieved entities with PageRank scores
            if msg["role"] == "assistant" and msg.get("entities"):
                entities = msg["entities"]
                if entities:
                    st.markdown("---")
                    st.caption(f"**Top-{min(len(entities), 8)} retrieved entities** (ranked by PageRank score)")
                    for i, e in enumerate(entities[:8], 1):
                        score = e.get('pagerank_score', 0)
                        etype = e.get('type', 'Unknown')
                        name = e.get('name', 'Unknown')
                        # Format score as percentage
                        score_pct = f"{score*100:.2f}%" if score > 0 else "—"
                        st.caption(f"**{i}.** {name} · _{etype}_ · PageRank: {score_pct}")

    # Process new message (from initial input or pill)
    if user_message:
        # Add to history and rerun to display properly
        st.session_state.messages.append({"role": "user", "content": user_message})

        with st.spinner("Searching..." if has_docs else "Thinking..."):
            try:
                if has_docs:
                    result = get_service().query(user_message, top_k=8)
                    answer = strip_citations(result.get("answer", "No answer found."))
                    entities = result.get("entities", [])
                    # If no entities found, mention it
                    if not entities and "Error" not in answer:
                        answer += "\n\n_Note: No specific entities were found in the knowledge graph for this query._"
                else:
                    answer = get_service().simple_chat(user_message)
                    entities = []
            except Exception as e:
                answer = f"Error: {str(e)}"
                entities = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "entities": entities
        })
        st.rerun()

    # Follow-up input
    if follow_up := st.chat_input("Ask a follow-up..."):
        st.session_state.messages.append({"role": "user", "content": follow_up})

        with st.spinner("Searching..." if has_docs else "Thinking..."):
            try:
                if has_docs:
                    result = get_service().query(follow_up, top_k=8)
                    answer = strip_citations(result.get("answer", "No answer found."))
                    entities = result.get("entities", [])
                    if not entities and "Error" not in answer:
                        answer += "\n\n_Note: No specific entities were found in the knowledge graph for this query._"
                else:
                    answer = get_service().simple_chat(follow_up)
                    entities = []
            except Exception as e:
                answer = f"Error: {str(e)}"
                entities = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "entities": entities
        })
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# GRAPH VISUALIZATION TAB
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    import plotly.graph_objects as go

    has_data = stats["nodes"] > 0
    col1, col2 = st.columns([1, 4])

    with col1:
        st.subheader("Controls")
        st.caption("Visualize knowledge graph with PageRank-scored nodes")

        if has_data:
            if st.button("Load graph", type="primary", icon=":material/refresh:", use_container_width=True):
                with st.spinner("Computing PageRank..."):
                    st.session_state.graph_data = get_service().get_graph_data()
                    st.rerun()
        else:
            st.button("Load graph", type="secondary", icon=":material/refresh:", use_container_width=True, disabled=True)

        max_n = st.slider("Top-k nodes", 10, 100, 40, disabled=not has_data, help="Show top-k nodes ranked by PageRank")

        gd = st.session_state.graph_data
        node_count = len(gd["nodes"]) if gd and gd.get("nodes") else 0
        edge_count = len(gd.get("edges", [])) if gd else 0

        st.metric("Nodes", node_count)
        st.metric("Edges", edge_count)

        # Top entities by PageRank
        if gd and gd.get("nodes"):
            st.subheader("Top by PageRank")
            top_nodes = sorted(gd["nodes"], key=lambda x: x.get("score", 0), reverse=True)[:5]
            for n in top_nodes:
                score = n.get('score', 0)
                st.caption(f":material/circle: {n['id']} ({score*100:.1f}%)")

    with col2:
        gd = st.session_state.graph_data

        # Create empty figure as default
        fig = go.Figure()
        fig.update_layout(
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
        )

        if gd and gd.get("nodes"):
            import networkx as nx

            nodes = sorted(gd["nodes"], key=lambda x: x.get("score", 0), reverse=True)[:max_n]
            node_ids = {n["id"] for n in nodes}

            G = nx.Graph()
            for n in nodes:
                G.add_node(n["id"], type=n.get("type", ""), score=n.get("score", 0))
            for e in gd.get("edges", []):
                if e["source"] in node_ids and e["target"] in node_ids:
                    G.add_edge(e["source"], e["target"])

            if len(G.nodes) >= 2:
                pos = nx.spring_layout(G, k=1.8, seed=42)
                colors = {
                    "Person": "#6366f1", "Organization": "#0891b2", "Location": "#22c55e",
                    "Event": "#f59e0b", "Discovery": "#8b5cf6", "Technology": "#3b82f6",
                    "Concept": "#ec4899", "Chemical Element": "#14b8a6",
                }

                edge_x, edge_y = [], []
                for u, v in G.edges():
                    edge_x.extend([pos[u][0], pos[v][0], None])
                    edge_y.extend([pos[u][1], pos[v][1], None])

                max_s = max((G.nodes[n]["score"] for n in G.nodes()), default=1) or 1

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=edge_x, y=edge_y, mode='lines',
                    line=dict(width=0.8, color='#666'), hoverinfo='none'
                ))
                fig.add_trace(go.Scatter(
                    x=[pos[n][0] for n in G.nodes()],
                    y=[pos[n][1] for n in G.nodes()],
                    mode='markers+text',
                    marker=dict(
                        size=[14 + 18*(G.nodes[n]["score"]/max_s) for n in G.nodes()],
                        color=[colors.get(G.nodes[n]["type"], "#888") for n in G.nodes()],
                        line=dict(width=1.5, color='white')
                    ),
                    text=list(G.nodes()),
                    textposition="top center",
                    textfont=dict(size=10),
                    hovertext=[f"{n}\n{G.nodes[n]['type']}" for n in G.nodes()],
                    hoverinfo='text'
                ))
                fig.update_layout(
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=500,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig, use_container_width=True)

                # Legend
                types_found = set(G.nodes[n]["type"] for n in G.nodes() if G.nodes[n]["type"])
                if types_found:
                    legend_html = " &nbsp;·&nbsp; ".join(
                        f'<span style="color:{colors.get(t, "#888")}">●</span> {t}'
                        for t in sorted(types_found)
                    )
                    st.markdown(f'<p style="font-size:0.85em; opacity:0.7">{legend_html}</p>', unsafe_allow_html=True)
            else:
                fig.add_annotation(text="Not enough connected nodes", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#666"))
                st.plotly_chart(fig, use_container_width=True)
        else:
            msg = "Upload documents to build knowledge graph" if not has_data else "Click 'Load graph' to visualize"
            fig.add_annotation(text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#666"))
            st.plotly_chart(fig, use_container_width=True)
