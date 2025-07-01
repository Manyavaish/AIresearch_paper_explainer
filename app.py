import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader  # ✅ fixed here
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_vertexai import ChatVertexAI

st.set_page_config(page_title="AI Research Paper Explainer", layout="centered")
st.title("📘 AI Research Paper Explainer")

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("⏳ Processing document...")

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)

    llm = ChatVertexAI(model_name="gemini-1.5-pro", temperature=0.3)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    st.success("✅ Paper processed!")

    if st.button("Generate Blog Summary"):
        st.subheader("🧩 Problem Statement")
        st.write(qa.run("What problem does this paper solve? Explain simply."))

        st.subheader("🛠️ Methodology")
        st.write(qa.run("Explain the methodology in simple terms."))

        st.subheader("🌟 Key Takeaways")
        st.write(qa.run("Summarize the results in 3-5 bullet points."))

        st.subheader("🧠 Conclusion")
        st.write(qa.run("Summarize the conclusion in a blog-style."))
