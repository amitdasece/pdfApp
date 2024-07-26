import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import chromadb
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from sympy import per
 
load_dotenv()
 
class IndexDoc:

    def index(self, db_name="chromadb"):
        try:
            loader = PyPDFLoader('D:/pdfproject/media/uploads/10840-001_JEBBHs7.pdf')
            docs = loader.load()
            print("Documents loaded successfully.")
        except Exception as e:
            print(f"Error loading documents: {e}")
            return
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=30,
        )
        chunks = text_splitter.split_documents(docs)
        print(f"Document split into {len(chunks)} chunks.")
        
        if db_name == "chromadb":
            vectorstore = Chroma.from_documents(
                documents=chunks, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db").as_retriever()
        else:
            vectorstore = FAISS.from_documents(
                chunks, embedding=OpenAIEmbeddings())
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
    # index.index()  # Index the document initially
    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        response = index.retrieve(user_input, db_type="chroma_db")
        print(response if response else "No relevant information found.")
