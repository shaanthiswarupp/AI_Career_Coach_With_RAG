from __future__ import annotations
import tempfile
from pathlib import Path
import docx
from langchain_community.document_loaders import PyPDFLoader


def read_uploaded_file(uploaded_file) -> str:
    if not uploaded_file:
        raise ValueError("No file uploaded.")

    uploaded_file.seek(0)
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")

    if suffix == ".docx":
        # python-docx reads the file-like buffer directly without temporary disk files
        doc = docx.Document(uploaded_file)
        full_text = []
        
        # Read paragraphs
        for para in doc.paragraphs:
            if para.text.strip():
                full_text.append(para.text.strip())
                
        # Also extract text inside tables (resumes frequently use tables)
        for table in doc.tables:
            for row in table.rows:
                row_data = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_data:
                    full_text.append(" | ".join(row_data))

        extracted_text = "\n\n".join(full_text).strip()
        if not extracted_text:
            raise ValueError("Could not extract any text from the .docx file.")
        return extracted_text

    if suffix == ".pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name

        try:
            loader = PyPDFLoader(temp_path)
            pages = loader.load()
            return "\n\n".join([page.page_content or "" for page in pages])
        finally:
            Path(temp_path).unlink(missing_ok=True)

    raise ValueError(f"Unsupported file type: {suffix}")










# from __future__ import annotations
# import tempfile

# from pathlib import Path   
# from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

# import docx2txt
# import docx

# def read_uploaded_file(uploaded_file) -> str:
        
        

#         if not uploaded_file:
#             raise ValueError("No file uploaded.")
        
#         uploaded_file.seek(0)
#         suffix = Path(uploaded_file.name).suffix.lower()# sssss.pdf => " .pdf "

#         if suffix == ".txt":
#             return uploaded_file.read().decode('utf-8', errors='ignore') # read()

#         # if suffix == ".docx":
#         #    text = docx2txt.process(uploaded_file)
#         #    return text # Docx2txtLoader(uploaded_file).load() => [Document(page_content=..., metadata=...)]
        
#         # if suffix == ".pdf":
#         #     reader = PyPDFLoader(uploaded_file)  # PyPDFLoader(uploaded_file) => PyPDFLoader(file_path=..., metadata=...)
#         #     text =[]

#         #     for page in reader.load():
#         #         text.append(page.page_content or "")

#         #     return "\n\n".join(text)    

            
#         # else:
#         #     raise ValueError(f"Unsupported file type: {suffix}")

#         # For formats needing a real file path (.pdf, .docx)

#         if suffix in [".pdf", ".docx"]:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
#                 tmp_file.write(uploaded_file.getbuffer())
#                 temp_path = tmp_file.name

#             try:
#                 if suffix == ".docx":
#                     return docx2txt.process(temp_path)

#                 if suffix == ".pdf":
#                     loader = PyPDFLoader(temp_path)
#                     pages = loader.load()
#                     return "\n\n".join([page.page_content or "" for page in pages])
#             finally:
#                 # Clean up the temp file from disk
#                 Path(temp_path).unlink(missing_ok=True)

#         raise ValueError(f"Unsupported file type: {suffix}")


