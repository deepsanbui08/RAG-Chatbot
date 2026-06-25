import streamlit as st

from pdf_processor import process_pdf
from vector_store import VectorStore
from rag_pipeline import answer_question


st.set_page_config(
    page_title="College Notes Chatbot",
    page_icon="📚"
)

st.title("📚 College Notes Chatbot")

st.write(
    "Upload your college notes and ask questions about them."
)

uploaded_file = st.file_uploader(
    "Upload PDF Notes",
    type=["pdf"]
)

if uploaded_file:

    if (
        "file_name" not in st.session_state
        or st.session_state.file_name != uploaded_file.name
    ):

        with st.spinner("Processing PDF..."):

            chunks = process_pdf(uploaded_file)

            vector_store = VectorStore()

            index = vector_store.create_index(
                chunks
            )

            st.session_state.file_name = (
                uploaded_file.name
            )

            st.session_state.chunks = chunks

            st.session_state.index = index

    st.success("PDF Loaded Successfully")

    st.write(
        f"Total Chunks: {len(st.session_state.chunks)}"
    )

    question = st.text_input(
        "Ask a question about your notes"
    )

    if question:

        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = answer_question(
                    question,
                    st.session_state.chunks,
                    st.session_state.index
                )

                st.subheader("Answer")

                st.write(answer)

            except Exception as e:

                st.error(
                    f"Error: {str(e)}"
                )