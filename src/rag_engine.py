import os
import shutil
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from langchain_core import documents
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import chromadb

load_dotenv()

db="chroma_db"

def get_llm(model:str = "llama3-70b-8192" , temperature: float = 0.2) -> ChatGroq:  
    
    return ChatGroq(model=model, temperature=temperature )

def get_embeddings(model_name: str = "hkunlp/instructor-xl") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=model_name)


#------------------->  1
def load_text_file( file_path: str , doc_type: str, source: str ) -> List[Document]:
   
   path = Path(file_path)
   text = path.read_text(encoding='utf-8', errors='ignore')
   return [ Document(page_content = text, metadata={"source": source, "doc_type": doc_type})]   
def create_documents( resume_text:str, jd_text:str ) -> List[Document]:

    resume_doc = Document(page_content=resume_text, metadata={"source": "resume", "doc_type": "resume"})
    jd_doc = Document(page_content=jd_text, metadata={"source": "job_description", "doc_type": "job_description"})


    return [resume_doc, jd_doc]

#------------------->  2
def split_documents( documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap , separators=["\n\n", "\n", " ", ""]  )

    splitted_documents = text_splitter.split_documents(documents)

    return splitted_documents






#------------------->  3 & 4
def create_vectorstore( chunks: List[Document], persist_directory="./chroma_db" ) -> Chroma:

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embeddings = get_embeddings()

    client = chromadb.PersistentClient(path=persist_directory)

    #             ========= vectorstore =========
    vectorstore = Chroma.from_documents(documents=chunks, embeddings=embeddings , client=client, collection_name="career_coach_collection")

    #vectorstore.persist()

    return vectorstore  






#------------------->  5
def retrieve_context_data( vectorstore: Chroma, query: str,  k: int = 3 ) -> List[Document]:


    docs = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    return docs, context


#------------------->  6
def run_career_coach( vectorstore, resume_text: str, jd_text: str, query: str, k: int = 3 ) -> str:
   
    #documents = create_documents(resume_text, jd_text) # Step 1: Create documents from resume and job description
   
    #chunks = split_documents(documents) # Step 2: Split documents into chunks
    
    #vectorstore = create_vectorstore(chunks)# Step 3: Create vectorstore from chunks
    
    retrieval_query = f""" Resume content and job description content relevant to this career coaching question:{query} """

    docs, context = retrieve_context_data(vectorstore, retrieval_query, k=6)# Step 4: Retrieve context data based on the query

    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template( """
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

    chain = prompt | llm | StrOutputParser 
    
    answer = chain.invoke( {"context": context, "query": query} )
    
    return answer , docs

#------------------->  7
def generate_complete_report(vectorstore, resume_text:str , jd_text:str ):

    query =  """ Analyze this resume against this job description. 
                Provide ATS-style score,
                skill match, missing skills,
                resume improvement suggestions,
                project suggestions, 
                and interview questions """
    return run_career_coach( vectorstore, resume_text, jd_text, query)
