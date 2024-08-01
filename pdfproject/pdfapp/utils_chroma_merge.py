import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import pdfplumber
import logging
from PyPDF2 import PdfMerger

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize ChromaDB
chroma_client = chromadb.Client()

# Example mapping of company names to PDF filenames
company_mapping = {
    "Intel": ["0000050863-24-000076-intel.pdf"],
    "AMD": ["0000002488-24-000056-amd.pdf"]
}

base_path = "D:/pdfproject/media/uploads/"

def merge_pdfs(company_name, pdf_list, output_path):
    """
    Merges multiple PDF files into one PDF file for a specific company.

    Parameters:
    company_name (str): Name of the company.
    pdf_list (list): List of PDF file names to be merged.
    output_path (str): Path to the output merged PDF file.
    """
    merger = PdfMerger()
    for pdf in pdf_list:
        full_path = os.path.join(base_path, pdf)
        if os.path.exists(full_path):
            print(f"Merging: {full_path}")
            merger.append(full_path)
        else:
            print(f"File not found: {full_path}")
    with open(output_path, 'wb') as output_file:
        merger.write(output_file)
    merger.close()
    print(f"PDFs for {company_name} merged successfully into {output_path}")

def process_pdf(input_pdf, company_name):
    try:
        with pdfplumber.open(input_pdf) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_text(all_text)
            embeddings = OpenAIEmbeddings()

            collection = chroma_client.get_or_create_collection(company_name)

            for i, doc in enumerate(docs):
                metadata = {
                    "company_name": company_name,
                    "chunk_number": i + 1,
                    "content": doc
                }
                vector = embeddings.embed_query(doc)
                collection.add(
                    ids=[f"{input_pdf}_chunk_{i}"],
                    embeddings=[vector],
                    metadatas=[metadata]
                )
    except Exception as e:
        logging.error(f"Error processing {input_pdf}: {e}")

def process_company_pdfs():
    for company_name, pdf_list in company_mapping.items():
        merged_output_path = os.path.join(base_path, f"{company_name}_merged.pdf")
        merge_pdfs(company_name, pdf_list, merged_output_path)
        process_pdf(merged_output_path, company_name)

def search_company(company_name, query):
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)

    collection = chroma_client.get_collection(company_name)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=1000
    )
    matches = results['metadatas'][0]

    if not matches:
        print(f"No results found for company: {company_name} with query: {query}")
        return []
    
    return [match['content'] for match in matches]

def main():
    process_company_pdfs()
    while True:
        company_name = input("Enter the company name to search: ")
        if company_name not in company_mapping:
            print(f"Company {company_name} not found. Please try again.")
            continue
        query = input(f"Enter the query to search within {company_name}'s documents: ")
        result_content = search_company(company_name, query)
        if result_content:
            print(f"Found relevant content for {company_name} with query '{query}':")
            for content in result_content:
                print(f"\nContent:\n{content}")
        else:
            print(f"No content found for {company_name} with query '{query}'.")

if __name__ == "__main__":
    main()
