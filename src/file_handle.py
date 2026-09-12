from __future__ import annotations
import tempfile

from pathlib import Path   
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader

import docx2txt


def read_uploaded_file(uploaded_file) -> str:

        if not uploaded_file:
            raise ValueError("No file uploaded.")
        
        suffix = Path(uploaded_file.name).suffix.lower()# sssss.pdf => " .pdf "

        if suffix == ".txt":
            return uploaded_file.read().decode('utf-8', errors='ignore') # read()

        # if suffix == ".docx":
        #    text = docx2txt.process(uploaded_file)
        #    return text # Docx2txtLoader(uploaded_file).load() => [Document(page_content=..., metadata=...)]
        
        # if suffix == ".pdf":
        #     reader = PyPDFLoader(uploaded_file)  # PyPDFLoader(uploaded_file) => PyPDFLoader(file_path=..., metadata=...)
        #     text =[]

        #     for page in reader.load():
        #         text.append(page.page_content or "")

        #     return "\n\n".join(text)    

            
        # else:
        #     raise ValueError(f"Unsupported file type: {suffix}")

        # For formats needing a real file path (.pdf, .docx)
        
        if suffix in [".pdf", ".docx"]:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                temp_path = tmp_file.name

            try:
                if suffix == ".docx":
                    return docx2txt.process(temp_path)

                if suffix == ".pdf":
                    loader = PyPDFLoader(temp_path)
                    pages = loader.load()
                    return "\n\n".join([page.page_content or "" for page in pages])
            finally:
                # Clean up the temp file from disk
                Path(temp_path).unlink(missing_ok=True)

        raise ValueError(f"Unsupported file type: {suffix}")


