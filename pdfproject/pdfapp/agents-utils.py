import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, Chroma
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.agents import Tool, AgentExecutor
from langchain.agents import initialize_agent

load_dotenv()

class IndexDoc:

    def index(self, db_name="chromadb"):
        loader = PyPDFLoader('D:/pdfproject/media/uploads/10840-001-chunks.pdf')
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=30,
        )
        chunks = text_splitter.split_documents(docs)
        if db_name == "chromadb":
            vectorstore = Chroma.from_documents(
                documents=chunks, embedding=OpenAIEmbeddings(), persist_directory="./chroma_db").as_retriever()
        else:
            vectorstore = FAISS.from_documents(
                chunks, embedding=OpenAIEmbeddings())
            vectorstore.save_local("./faiss_db")

        print("doc embedded and stored in chroma db")

    def retrieve(self, query, db_type="chroma_db"):
        llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        db = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings()
                    ) if db_type == "chroma_db" else FAISS.load_local("faiss_db", embeddings=OpenAIEmbeddings(), allow_dangerous_deserialization=True)
        retriever = db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5})
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define a simple retrieval function
        def retrieve_documents(query):
            docs = retriever.get_relevant_documents(query)
            return docs

        # Create a Tool for retrieval
        retrieval_tool = Tool(
            name="retrieve",
            func=retrieve_documents,
            description="Retrieve documents related to the query"
        )

        # Initialize the agent with the LLM and the retrieval tool
        agent = initialize_agent(
            tools=[retrieval_tool],
            llm=llm,
            agent_executor=AgentExecutor(),
            verbose=True
        )

        # Use the agent to handle the query
        response = agent.invoke(query)
        return response["answer"]

if __name__ == '__main__':
    index = IndexDoc()
    index.index()
    while True:
        user_input = input("Enter your command: ")
        if user_input.lower() == "exit":
            break
        print(index.retrieve(user_input, db_type="chroma_db"))
