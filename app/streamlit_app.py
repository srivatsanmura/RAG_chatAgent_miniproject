import streamlit as st
import pandas as pd

from rag.vectordb import get_vectordb
from rag.graph import build_graph
from rag.utils import prepare_visualization_data
from rag.agentMemory import AgentMemory


st.set_page_config(
    page_title="RAG Assistant",
    layout="wide"
)

st.title("Developer Docs RAG Assistant")

# Sidebar
st.sidebar.header("Observability Panel")

show_chunks = st.sidebar.checkbox(
    "Show retrieved chunks",
    value=True
)

show_scores = st.sidebar.checkbox(
    "Show score visualization",
    value=True
)


@st.cache_resource(show_spinner="Loading Graph...")
def load_graph():
    return build_graph()

@st.cache_resource(show_spinner="Loading Vector DB...")
def load_vectordb():
    return get_vectordb()

graph = load_graph()
vectordb = load_vectordb()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.memory = AgentMemory()


main_col, right_col = st.columns([2, 1])

with main_col:
    # Show chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                for s in msg["sources"]:
                    st.markdown(f"<small><i>- {s}</i></small>", unsafe_allow_html=True)

query = st.chat_input("Ask your question")


if query:

    state = {
        "query": query,
        "chat_history": f"System Summary: {st.session_state.memory.get_summary()}\nRecent Messages: {st.session_state.memory.buffer}",
        "vectordb": vectordb
    }

    with main_col:
        # Show query
        with st.chat_message("user"):
            st.write(query)

        # Show answer while streaming steps
        with st.chat_message("assistant"):
            with st.status("Processing query...", expanded=True) as status:
                st.write("Starting execution...")
                result_state = state.copy()
                for output in graph.stream(state):
                    for node_name, node_state in output.items():
                        st.write(f"Executed step: `{node_name}`")
                        # State updates can be merged back
                        if isinstance(node_state, dict):
                            result_state.update(node_state)
                status.update(label="Execution complete", state="complete", expanded=False)

            answer = result_state.get("answer", "")
            sources = result_state.get("sources", [])
            final_docs = result_state.get("compressed_docs", result_state.get("reranked_docs", []))

            st.write(answer)
            st.markdown("<b><u>Sources:</u></b>", unsafe_allow_html=True)
            st.markdown("<ul>", unsafe_allow_html=True)
            for s in sources:
                st.markdown(f"<li><small>{s}</small></li>", unsafe_allow_html=True)
            st.markdown("</ul>", unsafe_allow_html=True)

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )
    st.session_state.memory.add("user", query)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
    st.session_state.memory.add("assistant", answer)

    with right_col:
        # Observability panel
        if show_chunks:
            st.markdown("<b><u><small>Retrieved Chunks (Compressed)<small> </u></b>", unsafe_allow_html=True)
            for i, doc in enumerate(final_docs):
                with st.expander(f"Chunk {i+1}"):
                    st.markdown("<u>Sources:</u>", unsafe_allow_html=True)
                    st.markdown(f"<li><small><i>{doc['source']}</i></small></li>", unsafe_allow_html=True)
                 #   st.write(f"Vector Score: {doc.get('vector_score', 0):.4f}")
                    st.markdown("<u>Rerank Score:</u>", unsafe_allow_html=True)
                    st.markdown(f"<li><small><i>{doc.get('rerank_score', 0):.4f}</i></small></li>", unsafe_allow_html=True)
                    st.markdown("<u>Content:</u>", unsafe_allow_html=True)
                    st.markdown(f"<li><small><i>{doc['content']}</i></small></li>", unsafe_allow_html=True)

        if show_scores:

            st.markdown("<b><u><small>Score Visualization<small> </u></b>", unsafe_allow_html=True)

            df = prepare_visualization_data(final_docs)

            with st.container(border=True):
                tab1, tab2 = st.tabs(["Chart", "Dataframe"])
                with tab1:
                    st.subheader("Reranker Scores")
                    st.bar_chart(df.set_index("chunk_id")["rerank_score"])
                with tab2:
                    st.dataframe(df)
