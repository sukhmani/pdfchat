import streamlit as st
from app.extractor import extract_text_pymupdf
from app.embedder import embed_text, bert_model, bert_tokenizer
from app.retriever import retrieve_top_chunks
from app.generator import generate_answer

st.markdown("# PDFChat - A research tool")  
st.markdown("### A project by Jheyne de Oliveira Panta Cardeiro, Sumit Chahar, and Sukhmani Thukral")  
st.markdown("#### Under the guidance of Trevor Crosby")  


file = st.file_uploader("Upload a PDF", type="pdf")
if file:
    path = "data/temp.pdf"
    with open(path, "wb") as f:
        f.write(file.read())

    raw_text = extract_text_pymupdf(path)
    chunks = raw_text.split("\n\n")  # Simple chunking
    chunk_embeddings = [embed_text(c, bert_model, bert_tokenizer) for c in chunks]

    question = st.text_input("Ask a question")
    if question:
        top_chunks = retrieve_top_chunks(question, chunk_embeddings, chunks)
        answer = generate_answer(question, top_chunks)
        st.markdown(f"**Answer:** {answer}")

