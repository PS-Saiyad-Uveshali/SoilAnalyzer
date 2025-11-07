import io
import streamlit as st
import fitz  # PyMuPDF
from model import llm
from prompt import get_chain
from langchain_chroma import Chroma
from db import (
    make_embeddings,
    pages_to_documents,
    chunk_documents,
    build_or_update_vectorstore,
)

st.set_page_config(page_title="Soil Analyzer", layout="centered")
st.title("🌱 Soil Report Analyzer (with Vector DB)")

uploaded_file = st.file_uploader("Upload a Soil Test Report (PDF)", type=["pdf"])

if uploaded_file:
    with st.spinner("Reading PDF..."):
        # Keep a copy for naming the source
        file_bytes = uploaded_file.read()
        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        # Extract page-wise text so we can store metadata
        pages = []
        for i, page in enumerate(pdf):
            pages.append((i + 1, page.get_text()))

        # Full text for preview + for LLM analysis
        soil_text = "\n".join([text for _, text in pages])

    st.subheader("📄 Extracted Text Preview")
    st.text_area("PDF Content", soil_text, height=200)

    if st.button("📥 Index to Vector DB"):
        with st.spinner("Chunking & embedding..."):
            # Build embeddings using your Google API key in Streamlit secrets
            embeddings = make_embeddings(st.secrets["GOOGLE_API_KEY"])

            # Convert pages to LC Documents and chunk them
            docs = pages_to_documents(pages, source_name=uploaded_file.name)
            chunks = chunk_documents(docs)

            # Upsert into Chroma (persisted locally)
            _, retriever = build_or_update_vectorstore(
                chunks,
                persist_dir="chroma_db",
                embeddings=embeddings
            )

        st.success("✅ PDF indexed into the local Chroma DB (./chroma_db).")

        # Save retriever in session for reuse when analyzing
        st.session_state["retriever_ready"] = True

    if st.button("🔍 Analyze Soil Report"):
        if "retriever_ready" not in st.session_state:
            st.warning("Please click 'Index to Vector DB' first so I can retrieve context.")
        else:
            with st.spinner("Retrieving context & analyzing with Gemini..."):
                embeddings = make_embeddings(st.secrets["GOOGLE_API_KEY"])
                # Re-open the persisted Chroma and make a retriever
                vs = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
                retriever = vs.as_retriever(search_kwargs={"k": 6})

                # New retriever API (LC>=0.2): use invoke()
                query = "soil parameters, nutrients, pH, EC, organic matter, recommendations"
                retrieved_docs = retriever.invoke(query)

                context = "\n\n".join(
                    [f"[p.{d.metadata.get('page','?')}] {d.page_content[:800]}" for d in retrieved_docs]
                )

                chain = get_chain(llm)
                result = chain.invoke({"soil_text": soil_text, "context": context})

                st.success("✅ Analysis Completed!")
                st.subheader("🧪 Soil Quality")
                st.write(result.quality)

                st.subheader("🌾 Recommended Crops")
                st.write(", ".join(result.recommended_crops))

                st.subheader("💡 Suggestions")
                st.write(result.suggestions)

                st.subheader("🔎 Retrieved Context (top matches)")
                for d in retrieved_docs:
                    meta = d.metadata or {}
                    st.markdown(f"- **{meta.get('source','?')}**, page {meta.get('page','?')}")