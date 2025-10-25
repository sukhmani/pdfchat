import streamlit as st
from app.extractor import extract_with_pymupdf

st.set_page_config(layout="wide")
st.title("PDFChat")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_path = "data/temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())
    text = extract_with_pymupdf(temp_path)
    st.text_area("Extracted Text", text, height=400)
