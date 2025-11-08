import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load Groq API key
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Streamlit UI
st.set_page_config(page_title="Make It Asaan", layout="centered")
st.markdown("<h1 style='text-align:center;color:#4A90E2;'>📘 Make It Asaan</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center;color:gray;'>✨ Simplifying research papers and PDFs using AI ✨</h4>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your research paper (PDF)", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.info("⏳ Processing document...")

    # 1️⃣ Load & Split PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # 2️⃣ Create embeddings & vector DB
    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(chunks, embedding)
    retriever = db.as_retriever(search_kwargs={"k": 5})

    # 3️⃣ Initialize LLM
    llm = ChatGroq(model="llama3-70b-8192", temperature=0.3)

    # 4️⃣ Build retrieval chain manually (Runnable API)
    from langchain_core.prompts import PromptTemplate

    template = """Use the following retrieved context to answer the question clearly and simply.

    Context:
    {context}

    Question: {question}

    Answer:"""
    prompt = PromptTemplate.from_template(template)

    # Runnable chain: retrieve → format → call LLM → parse output
    retrieval_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    st.success("✅ Paper processed!")

    if st.button("Generate Blog Summary"):
        try:
            st.subheader("🧩 Problem Statement")
            st.write(retrieval_chain.invoke("What problem does this paper solve? Explain simply."))

            st.subheader("🛠️ Methodology")
            st.write(retrieval_chain.invoke("Explain the methodology in simple terms."))

            st.subheader("🌟 Key Takeaways")
            st.write(retrieval_chain.invoke("Summarize the results in 3–5 bullet points."))

            st.subheader("🧠 Conclusion")
            st.write(retrieval_chain.invoke("Summarize the conclusion in a blog-style."))

        except Exception as e:
            st.error("❌ Something went wrong while generating the summary.")
            st.exception(e)
