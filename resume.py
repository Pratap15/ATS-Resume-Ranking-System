import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to extract text from PDF
def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    # Encode job description and resumes
    job_description_embedding = model.encode(job_description, convert_to_tensor=True)
    resume_embeddings = model.encode(resumes, convert_to_tensor=True)
    
    # Calculate cosine similarity
    cosine_similarities = util.cos_sim(job_description_embedding, resume_embeddings).cpu().numpy().flatten()
    
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System (Enhanced with SBERT)")

# Job description input
st.header("Job Description")
job_description = st.text_area("Enter the job description")

# File uploader
st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files and job_description:
    st.header("Ranking Resumes")
    
    resumes = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        resumes.append(text)

    # Rank resumes
    scores = rank_resumes(job_description, resumes)

    # Display scores
    results = pd.DataFrame({"Resume": [file.name for file in uploaded_files], "Score": scores})
    results = results.sort_values(by="Score", ascending=False)
    
    st.write(results)
