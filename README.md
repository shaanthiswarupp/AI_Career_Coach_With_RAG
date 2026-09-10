# SSS AI Career Coach using Traditional RAG

This project is an end-to-end **traditional RAG application** for AI School of India members.

It uses:

- LangChain
- Groq Chat model
- HuggingFace Embeddings
- ChromaDB Vector Database
- Streamlit UI
- Resume + Job Description analysis

## Features

- Upload Resume as `.txt`, `.pdf`, or `.docx`
- Upload Job Description as `.txt`, `.pdf`, or `.docx`
- Build a RAG index from both documents
- Retrieve relevant context from Resume and JD
- Generate:
  - Resume match summary
  - Skill gap analysis
  - Resume improvement suggestions
  - Project recommendations
  - Interview preparation questions

## RAG Stages Covered

1. Document Loading
2. Chunking
3. Embeddings
4. Vector Database Storage
5. Query Embedding
6. Context Retrieval
7. LLM Answer Generation

## Setup

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

Windows PowerShell:

```bash
venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install requirements

```bash
pip install -r requirements.txt
```

### 4. Add Groq API key

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run app

```bash
streamlit run app.py
```

## Suggested Teaching Flow

1. Show the final app demo.
2. Explain Resume + JD as the knowledge base.
3. Explain document loading.
4. Explain chunking.
5. Explain embeddings.
6. Explain ChromaDB storage.
7. Explain query embedding and similarity search.
8. Explain LLM generation using retrieved context.
9. Show retrieved chunks in the UI.
10. Generate complete career report.

## Sample Files

Sample resume and job description are available in the `data/` folder.

## Important Note

For best compatibility with AI libraries, use Python 3.10, 3.11, or 3.12.
