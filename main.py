import os
import fitz  # PyMuPDF for PDF parsing
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from fastapi import FastAPI, File, UploadFile, HTTPException
from typing import List
from pydantic import BaseModel

app = FastAPI()

# Load models for embeddings and RAG
embedding_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
rag_tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
rag_retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="custom_faiss")
rag_model = RagSequenceForGeneration.from_pretrained("facebook/rag-token-nq", retriever=rag_retriever)

# FAISS index for storing PDF text embeddings
dimension = 384  # Dimension of the embeddings (from the model)
faiss_index = faiss.IndexFlatL2(dimension)

# Store text data and its corresponding embedding indices
pdf_text_data = []

class QuestionRequest(BaseModel):
    question: str

# Function to parse PDF and extract text
def extract_text_from_pdf(file_path: str) -> List[str]:
    doc = fitz.open(file_path)
    extracted_text = []
    for page in doc:
        extracted_text.append(page.get_text())
    return extracted_text

# Function to create embeddings and store them in FAISS
def index_pdf_content(pdf_text: List[str]):
    global pdf_text_data

    # Create embeddings for each sentence
    sentences = [sentence for page in pdf_text for sentence in page.split('.')]
    embeddings = embedding_model.encode(sentences)

    # Add the embeddings to the FAISS index
    faiss_index.add(np.array(embeddings, dtype=np.float32))

    # Store text for later retrieval
    pdf_text_data.extend(sentences)

# Upload PDF and process it
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File is not a PDF")

    # Save the PDF file locally (or you could upload to S3 here)
    pdf_path = f'./uploaded_pdfs/{file.filename}'
    with open(pdf_path, 'wb') as f:
        f.write(await file.read())

    # Extract and index the text content
    pdf_text = extract_text_from_pdf(pdf_path)
    index_pdf_content(pdf_text)

    return {"message": "PDF uploaded and indexed successfully"}

# Query RAG system
@app.post("/ask-question/")
async def ask_question(question: QuestionRequest):
    # Embed the user's question
    question_embedding = embedding_model.encode([question.question])

    # Retrieve relevant text chunks from FAISS
    D, I = faiss_index.search(np.array(question_embedding, dtype=np.float32), k=5)

    # Get the most relevant pieces of text
    relevant_texts = [pdf_text_data[i] for i in I[0]]

    # Combine the relevant text into a single context
    context = ' '.join(relevant_texts)

    # Generate an answer using RAG model
    inputs = rag_tokenizer(question.question, context, return_tensors="pt")
    generated_answers = rag_model.generate(input_ids=inputs['input_ids'])

    answer = rag_tokenizer.batch_decode(generated_answers, skip_special_tokens=True)[0]

    return {"answer": answer}

# Run the FastAPI server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)