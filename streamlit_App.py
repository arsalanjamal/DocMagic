import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import os

# Set up Streamlit page
st.set_page_config(page_title="Research Assistant", layout="wide")

st.markdown("""
## Research Assistant: Get insights, summaries, and citations from your research papers

This assistant helps you gain insights, retrieve key information, and generate citations from uploaded research papers. Upload your papers, ask questions, and get reliable, contextually relevant answers quickly.

### Key Features:
1. **Detailed Document Search**: Allows you to search within uploaded papers.
2. **Document Summarization**: Provides summaries for easier understanding.
3. **Citation Generator**: Generates citations for easy reference.

### Steps to Use:
1. **Enter Your Google API Key**: Obtain it at https://makersuite.google.com/app/apikey.
2. **Upload PDF Files**: Upload academic or research PDFs to process.
3. **Ask Your Question**: Get detailed answers based on the content.
""")

# API Key
api_key = st.text_input("Enter your Google API Key:", type="password", key="api_key_input")

# Helper Functions

# Extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Generate embeddings and store them in a vector store
def get_vector_store(text_chunks, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Load vector store and create conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question in detail based on the context provided. If information is not available, respond with "Information not found in the document."
    
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to answer user questions
def answer_user_question(user_question, api_key):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Answer: ", response["output_text"])

# Summarize the uploaded PDF document
def summarize_document(text_chunks):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    summary_prompt = """
    Summarize the following content briefly for a general understanding:\n
    Content:\n {content}\n
    Summary:
    """
    summary_template = PromptTemplate(template=summary_prompt, input_variables=["content"])
    chain = load_qa_chain(model, chain_type="map_reduce", prompt=summary_template)
    summaries = [chain({"input_documents": [chunk]}, return_only_outputs=True)["output_text"] for chunk in text_chunks]
    return "\n\n".join(summaries)

# Generate citation in a standard format
def generate_citation(doc_title, author, year):
    return f"{author} ({year}). *{doc_title}*. Retrieved from Document Genie Research Assistant."

# Main Streamlit App
def main():
    st.header("Research Assistant 📚")

    # User question input
    user_question = st.text_input("Ask a Question from the PDF Files", key="user_question")

    # Sidebar for uploading and processing PDFs
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True, key="pdf_uploader")
        if st.button("Submit & Process", key="process_button") and api_key:
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks, api_key)
                st.success("Processing complete!")
                
                # Summarize the document
                document_summary = summarize_document(text_chunks)
                st.subheader("Document Summary:")
                st.write(document_summary)
                
                # Example citation
                example_citation = generate_citation("Sample Document Title", "Author Name", "2023")
                st.subheader("Example Citation:")
                st.write(example_citation)
    
    # If question is entered, provide answer
    if user_question and api_key:
        answer_user_question(user_question, api_key)

if __name__ == "__main__":
    main()