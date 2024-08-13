import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
import pdfplumber
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Check if the index exists, if not, create it
index_name = 'pdf-chunks'
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

# Example mapping of company names to PDF filenames
company_mapping = {
    "Intel": ["0000050863-24-000076-intel.pdf"],
    "Micron": ["0001104659-24-045952-micron.pdf"]
}

base_path = "D:/pdfproject/media/uploads/"

def process_pdf(input_pdf, company_name):
    try:
        with pdfplumber.open(input_pdf) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"

            # Process the text content with LangChain
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            docs = text_splitter.split_text(all_text)
            embeddings = OpenAIEmbeddings()

            # Index the chunks in Pinecone
            for i, doc in enumerate(docs):
                metadata = {
                    "company_name": company_name,
                    "namespace": company_name,
                    "chunk_number": i + 1,
                    "content": doc
                }
                vector = embeddings.embed_query(doc)
                index.upsert([(f"{input_pdf}_chunk_{i}", vector, metadata)])
                response_out = index.describe_index_stats()
                print(response_out)
                logging.info(f"Indexed chunk {i + 1} for {input_pdf}")

    except Exception as e:
        logging.error(f"Error processing {input_pdf}: {e}")

def process_company_pdfs():
    for company_name, pdf_list in company_mapping.items():
        for pdf in pdf_list:
            pdf_path = os.path.join(base_path, pdf)
            if os.path.exists(pdf_path):
                logging.info(f"Processing {pdf_path}")
                process_pdf(pdf_path, company_name)
            else:
                logging.error(f"File {pdf_path} does not exist.")
    
    print("PDF processing and indexing completed for all companies.")

def search_company(company_name, query):
    # Query Pinecone for pages related to the company and containing the query
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(query)

    logging.info(f"Query vector for '{query}': {query_vector}")

    results = index.query(
        vector=query_vector,
        top_k=10,  # Adjust top_k based on your preference
        filter={
            "company_name": company_name
        }
    )

    logging.info(f"Query results: {results}")

    matches = results.get('matches', [])
    
    if not matches:
        print(f"No results found for company: {company_name} with query: {query}")
        return []
    
    # Extract the relevant content
    result_content = [match['metadata']['content'] for match in matches if 'metadata' in match]
    
    return result_content

def main():
    company_name = input("Enter the company name to search: ")
    query = input(f"Enter the query to search within {company_name}'s documents: ")
    result_content = search_company(company_name, query)
    
    if result_content:
        print(f"Found relevant content for {company_name} with query '{query}':")
        for content in result_content:
            print(f"\nContent:\n{content}")
    else:
        print(f"No content found for {company_name} with query '{query}'.")

if __name__ == "__main__":
    process_company_pdfs()
    main()
