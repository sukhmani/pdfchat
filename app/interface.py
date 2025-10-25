import streamlit as st
from app.extractor import extract_with_pymupdf, chunk_text
from app.embedder import find_best_match

st.title("PDFChat")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_path = "data/temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    raw_text = extract_with_pymupdf(temp_path)
    chunks = chunk_text(raw_text)

    st.text_area("Extracted Text", raw_text, height=300)

    question = st.text_input("Ask a question about the PDF")
    if question:
        answer = find_best_match(question, chunks)
        st.markdown(f"**Answer:** {answer}")

