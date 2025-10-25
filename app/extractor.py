import fitz  # PyMuPDF
import pdfplumber

def extract_text_pymupdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def extract_tables_pdfplumber(path):
    tables = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            tables.extend(page.extract_tables())
    return tables


