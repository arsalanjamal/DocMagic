import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # Updated import for FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
2. **Citation Generator**: Generates citations for easy reference.

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

# Highlight keywords in text
def highlight_keywords(text, query):
    highlighted_text = text.replace(query, f"<mark>{query}</mark>")
    return highlighted_text

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
    
    # Load the vector store (FAISS index) and allow dangerous deserialization
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search to find relevant documents
    docs = vector_store.similarity_search(user_question)
    
    # Get the conversational chain for answering the question
    chain = get_conversational_chain()
    
    # Get the response from the chain
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    # Highlight the keywords in the response
    highlighted_answer = highlight_keywords(response["output_text"], user_question)
    
    # Display the answer with highlighted keywords
    st.markdown(f"Answer: {highlighted_answer}", unsafe_allow_html=True)

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
                
                # Example citation
                example_citation = generate_citation("Sample Document Title", "Author Name", "2023")
                st.subheader("Example Citation:")
                st.write(example_citation)
    
    # If question is entered, provide answer
    if user_question and api_key:
        answer_user_question(user_question, api_key)

if __name__ == "__main__":
    main()
