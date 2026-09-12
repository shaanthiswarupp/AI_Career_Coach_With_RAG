import os
from typing import List, Tuple
from dotenv import load_dotenv

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    api_key = None
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Please add it to Streamlit Secrets or your .env file."
        )

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )


def get_embeddings() -> OpenAIEmbeddings:
    api_key = None
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    return OpenAIEmbeddings(api_key=api_key)


def create_documents(resume_text: str, jd_text: str) -> List[Document]:
    resume_doc = Document(
        page_content=resume_text,
        metadata={"source": "resume", "doc_type": "resume"},
    )
    jd_doc = Document(
        page_content=jd_text,
        metadata={"source": "job_description", "doc_type": "job_description"},
    )
    return [resume_doc, jd_doc]


def split_documents(
    documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return text_splitter.split_documents(documents)


def create_vectorstore(chunks: List[Document]) -> FAISS:
    if not chunks:
        raise ValueError("No text chunks found. Check that the files contain readable text.")
    
    # Creates an in-memory FAISS index directly serialized into Python state
    vectorstore = FAISS.from_documents(documents=chunks, embedding=get_embeddings())
    return vectorstore


def retrieve_context_data(
    vectorstore: FAISS, query: str, k: int = 4
) -> Tuple[List[Document], str]:
    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context


def run_career_coach(
    vectorstore: FAISS, resume_text: str, jd_text: str, query: str, k: int = 4
) -> Tuple[str, List[Document]]:
    retrieval_query = f"Resume and job description details regarding: {query}"
    docs, context = retrieve_context_data(vectorstore, retrieval_query, k=k)

    if not context.strip():
        context = "No relevant context found in the uploaded documents."

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
        """
You are an expert AI Career Coach for students, freshers and working professionals.
Use ONLY the given context from the resume and job description.
Do not invent skills, experience or job requirements.

CONTEXT:
{context}

USER QUESTION:
{query}

Give a clear, practical answer with these sections when relevant:
1. Current Match Summary
2. Strengths
3. Missing Skills / Gaps
4. Recommended Improvements
5. Suggested Projects
6. Interview Preparation Tips

Keep the answer simple, actionable and beginner-friendly.
"""
    )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query})
    return answer, docs


def generate_complete_report(vectorstore: FAISS, resume_text: str, jd_text: str):
    query = (
        "Analyze this resume against this job description. "
        "Provide ATS-style score, skill match, missing skills, "
        "resume improvement suggestions, project suggestions, and interview questions."
    )
    return run_career_coach(vectorstore, resume_text, jd_text, query)


# import os
# import shutil
# import tempfile
# from pathlib import Path
# from typing import List, Tuple
# from dotenv import load_dotenv

# import uuid
# import streamlit as st
# import chromadb
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_chroma import Chroma

# load_dotenv()


# def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
#     api_key = None
#     if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
#         api_key = st.secrets["OPENAI_API_KEY"]
#     else:
#         api_key = os.getenv("OPENAI_API_KEY")

#     if not api_key:
#         raise ValueError(
#             "OPENAI_API_KEY is not set. Please add it to Streamlit Secrets or your .env file."
#         )

#     return ChatOpenAI(
#         model=model,
#         temperature=temperature,
#         api_key=api_key,
#     )


# def get_embeddings() -> OpenAIEmbeddings:
#     api_key = None
#     if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
#         api_key = st.secrets["OPENAI_API_KEY"]
#     else:
#         api_key = os.getenv("OPENAI_API_KEY")

#     return OpenAIEmbeddings(api_key=api_key)


# def create_documents(resume_text: str, jd_text: str) -> List[Document]:
#     resume_doc = Document(
#         page_content=resume_text,
#         metadata={"source": "resume", "doc_type": "resume"},
#     )
#     jd_doc = Document(
#         page_content=jd_text,
#         metadata={"source": "job_description", "doc_type": "job_description"},
#     )
#     return [resume_doc, jd_doc]


# def split_documents(
#     documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200
# ) -> List[Document]:
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         separators=["\n\n", "\n", " ", ""],
#     )
#     return text_splitter.split_documents(documents)


# # def create_vectorstore(chunks: List[Document], persist_dir: str = "./chroma_db_store") -> Chroma:
# #     # Clear old collection on disk if re-indexing
# #     if os.path.exists(persist_dir):
# #         try:
# #             shutil.rmtree(persist_dir)
# #         except Exception:
# #             pass

# #     os.makedirs(persist_dir, exist_ok=True)
# #     client = chromadb.PersistentClient(path=persist_dir)

# #     # Automatically creates collection and adds embeddings
# #     vectorstore = Chroma.from_documents(
# #         documents=chunks,
# #         embedding=get_embeddings(),
# #         client=client,
# #         collection_name="career_coach_RAG",
# #     )
# #     return vectorstore



# def create_vectorstore(chunks: List[Document]) -> Chroma:
#     if not chunks:
#         raise ValueError("No text chunks found. Check that the uploaded files contain readable text.")

#     # Use a unique ephemeral path or EphemeralClient with Chroma's default settings
#     # to prevent locked SQLite database handles on Streamlit Cloud
#     client = chromadb.Client()  # Standard in-memory client managed by Chroma

#     # Unique collection name prevents collection collisions across button clicks
#     unique_collection = f"career_coach_{uuid.uuid4().hex[:8]}"

#     vectorstore = Chroma.from_documents(
#         documents=chunks,
#         embedding=get_embeddings(),
#         client=client,
#         collection_name=unique_collection,
#     )
#     return vectorstore


# def retrieve_context_data(    vectorstore: Chroma, query: str, k: int = 4) -> Tuple[List[Document], str]:
#     docs = vectorstore.similarity_search(query, k=k)
#     context = "\n\n".join([doc.page_content for doc in docs])
#     return docs, context


# def run_career_coach( vectorstore: Chroma, resume_text: str, jd_text: str, query: str, k: int = 4) -> Tuple[str, List[Document]]:

#     retrieval_query = f"Resume and job description details regarding: {query}"
#     docs, context = retrieve_context_data(vectorstore, retrieval_query, k=k)

#     if not context.strip():
#         context = "No relevant context found in the uploaded documents."

#     llm = get_llm()

#     prompt = ChatPromptTemplate.from_template(    """ You are an expert AI Career Coach for students, freshers and working professionals.
#                                                                     Use ONLY the given context from the resume and job description.
#                                                                 Do not invent skills, experience or job requirements.

#                                                                 CONTEXT:
#                                                                 {context}

#                                                                 USER QUESTION:
#                                                                 {query}

#                                                                 Give a clear, practical answer with these sections when relevant:
#                                                                 1. Current Match Summary
#                                                                 2. Strengths
#                                                                 3. Missing Skills / Gaps
#                                                                 4. Recommended Improvements
#                                                                 5. Suggested Projects
#                                                                 6. Interview Preparation Tips

#                                                                 Keep the answer simple, actionable and beginner-friendly.
#                                                                 """
#                                             )

#     chain = prompt | llm | StrOutputParser()
#     answer = chain.invoke({"context": context, "query": query})
#     return answer, docs


# def generate_complete_report(vectorstore: Chroma, resume_text: str, jd_text: str):
#     query = (
#         "Analyze this resume against this job description. "
#         "Provide ATS-style score, skill match, missing skills, "
#         "resume improvement suggestions, project suggestions, and interview questions."
#     )
#     return run_career_coach(vectorstore, resume_text, jd_text, query)





























# # import os
# # import shutil
# # import streamlit as st

# # from pathlib import Path
# # from typing import List
# # from dotenv import load_dotenv
# # from langchain_core import documents
# # from langchain_core.documents import Document
# # from langchain_core.prompts import ChatPromptTemplate
# # from langchain_core.output_parsers import StrOutputParser   
# # from langchain_text_splitters import RecursiveCharacterTextSplitter
# # from langchain_openai import ChatOpenAI
# # from langchain_huggingface import HuggingFaceEmbeddings
# # from langchain_openai import OpenAIEmbeddings
# # from langchain_chroma import Chroma

# # import chromadb

# # load_dotenv()

# # db="chroma_db"


# # def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
# #     api_key = None
# #     if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
# #         api_key = st.secrets["OPENAI_API_KEY"]
# #     else:
# #         api_key = os.getenv("OPENAI_API_KEY")

# #     if not api_key:
# #         raise ValueError(
# #             "OPENAI_API_KEY is not set. Please add it to Streamlit Secrets or your .env file."
# #         )

# #     return ChatOpenAI(
# #         model=model,
# #         temperature=temperature,
# #         api_key=api_key
# #     )

# # # def get_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbeddings:
# # #     return HuggingFaceEmbeddings(model_name=model_name)

# # def get_embeddings():
# #     return OpenAIEmbeddings()

# # #------------------->  1
# # def load_text_file( file_path: str , doc_type: str, source: str ) -> List[Document]:
   
# #    path = Path(file_path)
# #    text = path.read_text(encoding='utf-8', errors='ignore')
# #    return [ Document(page_content = text, metadata={"source": source, "doc_type": doc_type})]   

# # def create_documents( resume_text:str, jd_text:str ) -> List[Document]:

# #     resume_doc = Document(page_content=resume_text, metadata={"source": "resume", "doc_type": "resume"})
# #     jd_doc = Document(page_content=jd_text, metadata={"source": "job_description", "doc_type": "job_description"})


# #     return [resume_doc, jd_doc]

# # #------------------->  2
# # def split_documents( documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:

# #     text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap , separators=["\n\n", "\n", " ", ""]  )

# #     splitted_documents = text_splitter.split_documents(documents)

# #     return splitted_documents


# # #------------------->  3 & 4
# # def create_vectorstore( chunks):

# #     # if os.path.exists(persist_directory):
# #     #     shutil.rmtree(persist_directory)    

# #     # client = chromadb.PersistentClient(path=persist_directory)    
# #     client = chromadb.EphemeralClient()

# #     #========= vectorstore =========
# #     vectorstore = Chroma.from_documents(documents = chunks, embedding = get_embeddings() , client=client, collection_name="career_coach_RAG",)
# #     return vectorstore



# # #------------------->  5
# # def retrieve_context_data( vectorstore: Chroma, query: str,  k: int = 4 ) -> List[Document]:


# #     docs = vectorstore.similarity_search(query, k=k)

# #     context = "\n\n".join([doc.page_content for doc in docs])

# #     return docs, context


# # #------------------->  6
# # def run_career_coach( vectorstore, resume_text: str, jd_text: str, query: str, k: int = 3 ) -> str:
   
# #     #documents = create_documents(resume_text, jd_text) # Step 1: Create documents from resume and job description
   
# #     #chunks = split_documents(documents) # Step 2: Split documents into chunks
    
# #     #vectorstore = create_vectorstore(chunks)# Step 3: Create vectorstore from chunks
    
# #     retrieval_query = f""" Resume content and job description content relevant to this career coaching question:{query} """

# #     docs, context = retrieve_context_data(vectorstore, retrieval_query, k=4)# Step 4: Retrieve context data based on the query

# #     if not context.strip():
# #         context = "No relevant context found in the uploaded documents."
    
# #     llm = get_llm()
    
# #     prompt = ChatPromptTemplate.from_template( """
# #                                                     You are an expert AI Career Coach for students, freshers and working professionals.
# #                                                     Use ONLY the given context from the resume and job description.
# #                                                     Do not invent skills, experience or job requirements.

# #                                                     CONTEXT:
# #                                                     {context}

# #                                                     USER QUESTION:
# #                                                     {query}

# #                                                     Give a clear, practical answer with these sections when relevant:
# #                                                     1. Current Match Summary
# #                                                     2. Strengths
# #                                                     3. Missing Skills / Gaps
# #                                                     4. Recommended Improvements
# #                                                     5. Suggested Projects
# #                                                     6. Interview Preparation Tips

# #                                                     Keep the answer simple, actionable and beginner-friendly.
# #                                                     """
# #                                             )      

# #     chain = prompt | llm | StrOutputParser()
    
# #     answer = chain.invoke( {"context": context, "query": query} )
    
# #     return answer , docs

# # #------------------->  7
# # def generate_complete_report(vectorstore, resume_text:str , jd_text:str ):

# #     query =  """ Analyze this resume against this job description. 
# #                 Provide ATS-style score,
# #                 skill match, missing skills,
# #                 resume improvement suggestions,
# #                 project suggestions, 
# #                 and interview questions """
# #     return run_career_coach( vectorstore, resume_text, jd_text, query)
