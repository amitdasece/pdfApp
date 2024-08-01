import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
import logging
from PyPDF2 import PdfMerger

# Set up logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists, if not, create it
index_name = 'company-docs'
custom_url = 'https://document-index-pine-kh0uhge.svc.aped-4627-b74a.pinecone.io'
if index_name not in [idx.name for idx in pc.list_indexes().indexes]:
    pc.create_index(
        name=index_name,
        dimension=1536,  # Adjust dimension as per your use case
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Get the Pinecone index
index = pc.Index(index_name)
index._endpoint = custom_url

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
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            docs = text_splitter.split_text(all_text)
            embeddings = OpenAIEmbeddings()
            for i, doc in enumerate(docs):
                metadata = {
                    "id": f"{input_pdf}_chunk_{i}",
                    "company_name": company_name,
                    "chunk_number": i + 1,
                    "namespace": company_name,
                    "content": doc,
                    "values": extract_values(doc)  # Add extracted values to metadata
                }
                vector = embeddings.embed_query(doc)
                if vector:
                    index.upsert([(metadata["id"], vector, metadata)])
                else:
                    logging.error(f"Empty vector for {metadata['id']}")
    except Exception as e:
        logging.error(f"Error processing {input_pdf}: {e}")

def extract_values(text):
    """
    Extract relevant values from the text.
    Modify this function based on what specific values you want to extract.
    """
    # Example: Extracting all numbers from the text
    import re
    values = re.findall(r'\d+', text)
    return values

def process_company_pdfs():
    for company_name, pdf_list in company_mapping.items():
        merged_output_path = os.path.join(base_path, f"{company_name}_merged.pdf")
        merge_pdfs(company_name, pdf_list, merged_output_path)
        process_pdf(merged_output_path, company_name)

def search_company(company_name, query):
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)
    results = index.query(
        vector=query_vector,
        top_k=1,
        filter={
            "company_name": company_name
        }
    )
    matches = results.get('matches', [])
    if not matches:
        print(f"No results found for company: {company_name} with query: {query}")
        return []
    return [match['metadata']['content'] for match in matches if 'metadata' in match]

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
