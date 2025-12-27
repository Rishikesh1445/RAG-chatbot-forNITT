import os
from pypdf import PdfReader

PDF_DIR = "../data/pdf_raw"
OUTPUT_DIR = "../data/pdf"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n--- Page {i+1} ---\n{text}"

    return full_text.strip()

if __name__ == "__main__":
    for file in os.listdir(PDF_DIR):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, file)
            text = extract_pdf_text(pdf_path)

            if len(text) > 500:
                output_file = file.replace(".pdf", ".txt")
                with open(f"{OUTPUT_DIR}/{output_file}", "w", encoding="utf-8") as f:
                    f.write(text)
