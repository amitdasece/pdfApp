import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from pinecone import Pinecone as PineconeClient, Index, ServerlessSpec

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = PineconeClient(
    api_key='f47fbfc4-a053-4652-8826-f77f5b1bdb62'
)

class IndexDoc:
    def index(self, db_name="pinecone_db"):
        try:
            loader = PyPDFLoader('D:/pdfproject/media/uploads/Intel_merged.pdf')
            docs = loader.load()
            print("Documents loaded successfully.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return
        
        # Initial split into main chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        main_chunks = text_splitter.split_documents(docs)
        sub_chunks = text_splitter.split_documents(main_chunks)
        
        print(f"Document split into {len(sub_chunks)} subchunks.")
        
        # Initialize the Pinecone index
        index_name = "pdf-chunks"
        dimension = 1536  # OpenAI Embeddings dimension
        if index_name not in pc.list_indexes().names():
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=os.getenv('PINECONE_ENVIRONMENT')
                )
            )
        
        index = Index(index_name, host=f"{index_name}.svc.{os.getenv('PINECONE_ENVIRONMENT')}.pinecone.io")
        
        # Store documents in Pinecone
        embeddings = OpenAIEmbeddings()
        vectorstore = LangchainPinecone.from_documents(
            documents=sub_chunks, embedding=embeddings, index_name=index_name, text_key="text"
        )
        
        print("Document embedded and stored in the Pinecone database.")
    
    def retrieve(self, query, db_type="pinecone_db"):
        try:
            llm = ChatOpenAI()
            index_name = "pdf-chunks"
            custom_url = 'https://pdf-chunks-gpu38ye.svc.aped-4627-b74a.pinecone.io'
            # index = Index(index_name, host=f"{index_name}.svc.{os.getenv('PINECONE_ENVIRONMENT')}.pinecone.io")
            index = Index(index_name, host=custom_url)
            
            embeddings = OpenAIEmbeddings()
            retrv = LangchainPinecone(index=index, embedding=embeddings, text_key="text")
            docs = retrv.similarity_search(query, k=5)
            print(f"Retrieved {len(docs)} relevant documents.")
            
            memory = ConversationBufferMemory(llm=llm, memory_key="chat_history", return_messages=True)
            chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retrv, memory=memory)
            res = chain.invoke(query)
            return res['answer']
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None
 
if __name__ == '__main__':
    index = IndexDoc()
    index.index()  # Index the document initially
    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        response = index.retrieve(user_input, db_type="pinecone_db")
        print(response if response else "No relevant information found.")
