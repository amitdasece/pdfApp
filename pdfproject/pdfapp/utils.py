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
 
def format_response(text):
    # Format the text to preserve newlines for better readability
    return text.replace('\n', '<br>')
 
 
class IndexDoc:
 
    def index(self, db_name="chromadb"):
        loader = PyPDFLoader('D:/pdfproject/media/uploads/NVIDIA-2024-Annual-Report.pdf')
        docs = loader.load()
        text_spilter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=30,
        )
        chunks = text_spilter.split_documents(docs)
        if db_name == "chromadb":
            vectorstore = Chroma.from_documents(
                documents=chunks, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db").as_retriever()
        else:
            vectorstore = FAISS.from_documents(
                chunks, embedding=OpenAIEmbeddings())
            vectorstore.save_local("./faiss_db")
 
        print("doc embedded and stored in chroma db")
 
    def retrieve(self, query, db_type="chroma_db"):
        llm = ChatOpenAI()
 
        db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
                    ) if db_type == "chroma_db" else FAISS.load_local("faiss_db", embeddings=OpenAIEmbeddings(),allow_dangerous_deserialization=True)
        retrv = db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5})
        docs = retrv.get_relevant_documents(query)
        memory = ConversationBufferMemory(llm=llm,memory_key="chat_history",return_messages=True)
        # chain = RetrievalQA.from_chain_type(llm=llm, retriever=retrv)
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retrv,memory=memory)
        res = chain.invoke(query)
        return format_response(res['answer'])
 
 
if __name__ == '__main__':
    index = IndexDoc()
    index.index()
    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        print(index.retrieve(user_input,db_type="chroma_db"))
