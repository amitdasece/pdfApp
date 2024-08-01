import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_openai import ChatOpenAI
import chromadb

load_dotenv()

class IndexDoc:

    def index(self, db_name="chromadb"):
        try:
            loader = PyPDFLoader('D:/pdfproject/media/uploads/10840-001-chunks.pdf')
            docs = loader.load()
            print("Documents loaded successfully.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return
        
        # Initial split into 11 main chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1028,
            chunk_overlap=150,
        )
        main_chunks = text_splitter.split_documents(docs)
        main_chunks = [main_chunks[i::11] for i in range(11)]
        
        # Split each main chunk into 33 subchunks
        sub_chunks = []
        for main_chunk in main_chunks:
            for chunk in main_chunk:
                sub_chunks += text_splitter.split_documents([chunk])
        
        # Ensure we get the exact required number of subchunks (11 main * 33 sub = 363 subchunks)
        if len(sub_chunks) > 33:
            sub_chunks = sub_chunks[:33]
        elif len(sub_chunks) < 33:
            print(f"Warning: Only {len(sub_chunks)} subchunks were created, but 33 were expected.")
        
        print(f"Document split into {len(sub_chunks)} subchunks.")
        
        if db_name == "chromadb":
            vectorstore = Chroma.from_documents(
                documents=sub_chunks, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db").as_retriever()
        else:
            vectorstore = FAISS.from_documents(
                sub_chunks, embedding=OpenAIEmbeddings())
            vectorstore.save_local("./faiss_db")
        
        print("Document embedded and stored in the database.")
 
    def retrieve(self, query, db_type="chroma_db"):
        try:
            llm = ChatOpenAI()
            if db_type == "chroma_db":
                db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings())
            else:
                db = FAISS.load_local("faiss_db", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            retrv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            
            tools = [
                Tool(
                    name="Document Retrieval",
                    func=lambda query: retrv.get_relevant_documents(query),
                    description="Retrieve relevant documents based on the query"
                )
            ]
            
            agent = initialize_agent(
                tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
            )
            
            response = agent.run(query)
            return response
        
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return None
 
if __name__ == '__main__':
    index = IndexDoc()
    # index.index()  # Index the document initially
    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        response = index.retrieve(user_input, db_type="chroma_db")
        print(response if response else "No relevant information found.")




