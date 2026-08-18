from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='generate a 5 facts about {topic}', 
    input_variables=['topic']
)

parser=StrOutputParser() 

chain=prompt|model|parser

print(chain.invoke({'topic':'cricket'}))