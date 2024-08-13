import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI

load_dotenv()

class IndexDoc:

    def index(self, file_path, db_name="chromadb"):
        try:
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            print("Documents loaded successfully.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
        )
        chunks = text_splitter.split_documents(docs)
        
        persist_directory = f"./chroma_db_{os.path.basename(file_path).split('.')[0]}"
        
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=OpenAIEmbeddings(), persist_directory=persist_directory).as_retriever()
        
        print(f"Document embedded and stored in the database {persist_directory}.")

    def retrieve(self, query, db_name="chroma_db"):
        try:
            llm = ChatOpenAI()
            db = Chroma(persist_directory=db_name, embedding_function=OpenAIEmbeddings())
            retrv = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})
            docs = retrv.get_relevant_documents(query)
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
    
    # Index different PDF files
    # index.index('D:/pdfproject/media/uploads/AIRBNB.pdf', db_name="chroma_db_airbnb")
    # index.index('D:/pdfproject/media/uploads/intel.pdf', db_name="chroma_db_intel")
    # index.index('D:/pdfproject/media/uploads/NVIDIA.pdf', db_name="chroma_db_nvidia")

    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        response = index.retrieve(user_input, db_name="chroma_db_intel")  # specify the correct db_name here
        print(response if response else "No relevant information found.")
