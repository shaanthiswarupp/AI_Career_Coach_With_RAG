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
def create_vectorstore( chunks: List[Document], persist_directory: str = db ) -> Chroma:

    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)

    embeddings = get_embeddings()

    #             ========= vectorstore =========
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory)

    #vectorstore.persist()

    return vectorstore  

#------------------->  5
def retrieve_context_data( vectorstore: Chroma, query: str,  k: int = 3 ) -> List[Document]:


    docs = vectorstore.similarity_search(query, k=k)

    context = "\n\n".join([doc.page_content for doc in docs])

    return docs, context


#------------------->  6
def run_career_coach( resume_text: str, jd_text: str, query: str, chunk_size: int = 1000, chunk_overlap: int = 200, k: int = 3 ) -> str:
   
    documents = create_documents(resume_text, jd_text) # Step 1: Create documents from resume and job description
   
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap) # Step 2: Split documents into chunks
    
    vectorstore = create_vectorstore(chunks)# Step 3: Create vectorstore from chunks
    
    docs, context = retrieve_context_data(vectorstore, query, k=k)# Step 4: Retrieve context data based on the query

    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_template( """ You are a career coach. 
                                                You have been given a resume and a job description. 
                                                Your task is to provide a detailed analysis of how well the resume matches the job description.
                                                Use the following context to answer the question: {context}
                                                Resume: {resume}
                                                Job Description: {jd}
                                                Question: {query}  """ 
                                            )     
    
    query = f"Analyze the resume content and job content relavant to career coaching question"   

    chain = prompt | llm | StrOutputParser 
    
    answer = chain.invoke( {"context": context, "resume": resume_text, "jd": jd_text, "query": query} )
    
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
