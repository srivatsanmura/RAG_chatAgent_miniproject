import streamlit as st
import pandas as pd

from rag.vectordb import load_vectordb
from rag.graph import build_graph
from rag.utils import prepare_visualization_data


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


if "graph" not in st.session_state:
    st.session_state.graph = build_graph()

if "vectordb" not in st.session_state:
    st.session_state.vectordb = load_vectordb()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


query = st.chat_input("Ask your question")


if query:

    state = {

        "query": query,
        "chat_history": st.session_state.chat_history,
        "vectordb": st.session_state.vectordb

    }

    result = st.session_state.graph.invoke(state)

    answer = result["answer"]

    reranked_docs = result["reranked_docs"]

    st.session_state.chat_history.append(
        {"role": "user", "content": query}
    )

    st.session_state.chat_history.append(
        {"role": "assistant", "content": answer}
    )

    # Show answer
    with st.chat_message("assistant"):

        st.write(answer)

        st.write("### Sources")

        for s in result["sources"]:
            st.write("-", s)

    # Observability panel
    if show_chunks:

        st.write("## Retrieved Chunks")

        for i, doc in enumerate(reranked_docs):

            with st.expander(f"Chunk {i+1}"):

                st.write("Source:", doc["source"])

                st.write(
                    f"Vector Score: {doc['vector_score']:.4f}"
                )

                st.write(
                    f"Rerank Score: {doc['rerank_score']:.4f}"
                )

                st.write(doc["content"])


    if show_scores:

        st.write("## Score Visualization")

        df = prepare_visualization_data(reranked_docs)

        st.dataframe(df)

        st.write("### Vector Similarity Scores")

        st.bar_chart(
            df.set_index("chunk_id")["vector_score"]
        )

        st.write("### Reranker Scores")

        st.bar_chart(
            df.set_index("chunk_id")["rerank_score"]
        )


# Show chat history
for msg in st.session_state.chat_history:

    with st.chat_message(msg["role"]):
        st.write(msg["content"])