from typing import List, Tuple
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # ⬅️ updated import
from langchain_core.documents import Document

def make_embeddings(api_key: str):
    return GoogleGenerativeAIEmbeddings(model="text-embedding-004", google_api_key=api_key)

def make_text_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )

def pages_to_documents(pages: List[Tuple[int, str]], source_name: str) -> List[Document]:
    return [
        Document(page_content=text, metadata={"source": source_name, "page": page_no})
        for page_no, text in pages
    ]

def chunk_documents(docs: List[Document]):
    splitter = make_text_splitter()
    return splitter.split_documents(docs)

def build_or_update_vectorstore(docs: List[Document], persist_dir: str, embeddings):
    """
    If DB exists, add new docs; else create. Returns (vectorstore, retriever).
    """
    try:
        vs = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        vs.add_documents(docs)
    except Exception:
        vs = Chroma.from_documents(docs, embeddings, persist_directory=persist_dir)
    retriever = vs.as_retriever(search_kwargs={"k": 6})
    return vs, retriever
