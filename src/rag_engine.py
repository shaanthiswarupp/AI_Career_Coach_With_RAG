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


def split_documents(  documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:

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


def retrieve_context_data(    vectorstore: FAISS, query: str, k: int = 4) -> Tuple[List[Document], str]:

    docs = vectorstore.similarity_search(query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
    return docs, context

def extract_candidate_name(resume_text: str) -> str:
    """Extracts the candidate's full name from the uploaded resume text."""
    if not resume_text or not resume_text.strip():
        return ""

    llm = get_llm(temperature=0.0)    
    prompt = ChatPromptTemplate.from_template(
                                                        """Extract only the candidate's full name from the following resume text header.
                                                Return ONLY the name (2 to 4 words). Do not include words like 'Resume', 'CV', email addresses, phone numbers, or any punctuation.
                                                If no clear person name is found, return 'Candidate'.

                                                RESUME TEXT:
                                                {resume_snippet}
                                                """
                                            )
    
    chain = prompt | llm | StrOutputParser()
    # Pass only the first 1000 characters where names are located
    name = chain.invoke({"resume_snippet": resume_text[:1000]}).strip()
    return name if name else ""



def run_career_coach( vectorstore: FAISS, resume_text: str, jd_text: str, query: str, k: int = 4) -> Tuple[str, List[Document]]:
    
    retrieval_query = f"Resume and job description details regarding: {query}"
    docs, context = retrieve_context_data(vectorstore, retrieval_query, k=k)

    if not context.strip():
        context = "No relevant context found in the uploaded documents."

    candidate_name = extract_candidate_name(resume_text)

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
                                                
                                                Format your response clearly with these sections when relevant:
                                                Candidate Name: {candidate_name}

                                                Give a clear, practical answer with these sections when relevant:
                                            
                                                Answer to u r Question {candidate_name}
                                                1. Current Match Summary
                                                2. Strengths
                                                3. Missing Skills / Gaps
                                                4. Recommended Improvements
                                                5. Suggested Projects
                                                6. Interview Preparation Tips
                                                7. Suggestions to Improve

                                                Keep the answer simple, actionable and beginner-friendly.
                                                """
                                            )

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "query": query, "candidate_name":candidate_name} )
    return answer, docs


def generate_complete_report(vectorstore: FAISS, resume_text: str, jd_text: str):
    query = (
        "Analyze this resume against this job description. "        
        "Provide ATS-style score, skill match, missing skills, "
        "resume improvement suggestions, project suggestions, and interview questions."
        "where to improve exactly more and focus on"
    )
    return run_career_coach(vectorstore, resume_text, jd_text, query)

