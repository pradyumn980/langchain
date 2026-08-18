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

template=PromptTemplate(
    template='Generate a detail report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='Generate a 5 line summary on this report {text}',
    input_variables=['text']
)

parser=StrOutputParser()

chain=template|model|parser|template2|model|parser

result=chain.invoke({'topic':'liner regresion'})

print(result)
