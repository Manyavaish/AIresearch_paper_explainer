import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]


st.set_page_config(page_title="Make It Asaan", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📘 Make It Asaan</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>✨ Simplifying research papers and PDFs using AI ✨</h4>", unsafe_allow_html=True)

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

    
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

   
    retriever = db.as_retriever(search_kwargs={"k": 5})
    document_chain = create_stuff_documents_chain(llm)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    st.success("✅ Paper processed!")

    if st.button("Generate Blog Summary"):
        try:
            st.subheader("🧩 Problem Statement")
            st.write(qa_chain.invoke({"input": "What problem does this paper solve? Explain simply."})["answer"])

            st.subheader("🛠️ Methodology")
            st.write(qa_chain.invoke({"input": "Explain the methodology in simple terms."})["answer"])

            st.subheader("🌟 Key Takeaways")
            st.write(qa_chain.invoke({"input": "Summarize the results in 3-5 bullet points."})["answer"])

            st.subheader("🧠 Conclusion")
            st.write(qa_chain.invoke({"input": "Summarize the conclusion in a blog-style."})["answer"])

        except Exception as e:
            st.error("❌ Something went wrong while generating the summary.")
            st.exception(e)
