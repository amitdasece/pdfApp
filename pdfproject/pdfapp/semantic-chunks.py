from langchain.chains 
import TextChunker from langchain.llms 
import OpenAI 
# Initialize the LLM 
llm = OpenAI(model="gpt-4") 
# Create a TextChunker instance 
chunker = TextChunker(llm=llm) 
# Chunk some text 
text = "Your long body of text goes here." 
chunks = chunker.chunk(text)