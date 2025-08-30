import streamlit as st
import pdfplumber
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
import pickle

# -------------------------------
# PDF Parsing & Chunking
# -------------------------------
def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "
    return text

def split_text_into_chunks(text, chunk_size=500, overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    return docs

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(page_title="PDF RAG Chatbot", layout="centered")
st.title("📄 Optimized PDF RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Prepare FAISS index filename
    index_filename = uploaded_file.name.replace(".pdf", "_faiss")
    emb_filename = uploaded_file.name.replace(".pdf", "_docs.pkl")

    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 1️⃣ Check if FAISS index exists
    if os.path.exists(index_filename) and os.path.exists(emb_filename):
        with st.spinner("Loading existing embeddings and FAISS index..."):
            with open(emb_filename, "rb") as f:
                docs = pickle.load(f)
            # Allow dangerous deserialization because we trust our own pickle
            vectorstore = FAISS.load_local(
                index_filename,
                embedding_model,
                allow_dangerous_deserialization=True
            )
        st.success("Loaded cached embeddings!")
    else:
        with st.spinner("Processing PDF and generating embeddings..."):
            # Extract text
            text = extract_text_from_pdf(uploaded_file)
            if not text.strip():
                st.error("No text could be extracted from the PDF.")
            else:
                # Split into chunks
                docs = split_text_into_chunks(text)
                # Generate embeddings and build FAISS
                vectorstore = FAISS.from_documents(docs, embedding_model)
                # Save for future use
                vectorstore.save_local(index_filename)
                with open(emb_filename, "wb") as f:
                    pickle.dump(docs, f)
                st.success("PDF processed and embeddings cached!")

    # -------------------------------
    # 2️⃣ Load HuggingFace LLM (flan-t5-base)
    # -------------------------------
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=100
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # -------------------------------
    # 3️⃣ Define a concise-answer prompt
    # -------------------------------
    concise_prompt = PromptTemplate(
        template="""
Answer the question using the context. If the answer is not present, say "I don't know".
Keep it to 1–2 sentences. Do NOT repeat the question or context.

Context:
{context}

Question:
{question}

Answer:""",
        input_variables=["context", "question"]
    )

    # -------------------------------
    # 4️⃣ Setup RetrievalQA with the concise prompt
    # -------------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff",
        chain_type_kwargs={"prompt": concise_prompt}
    )

    # -------------------------------
    # 5️⃣ Ask user question
    # -------------------------------
    user_question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer"):
        if user_question.strip():
            with st.spinner("Generating answer..."):
                answer = qa_chain.run(user_question)
            st.subheader("Answer:")
            st.write(answer)
        else:
            st.warning("Please type a question first.")
