import fitz  # PyMuPDF

def extract_text(path):
    text = ''
    with fitz.open(path) as doc:
        for page in doc:
            text += page.get_text()
    return text