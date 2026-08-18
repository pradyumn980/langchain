from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)

template=PromptTemplate(
    template='Generate notes on {notes}',
    input_variables=['notes']
)

template2=PromptTemplate(
    template='Generate a quiz on {quiz}',
    input_variables=['quiz']
)

template3=PromptTemplate(
    template='Generate a combined document on  {notes} and {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallelChain=RunnableParallel({
    'notes': template|model|parser,
    'quiz':template2|model|parser

})

merge=template3|model|parser

chain=parallelChain|merge

result=chain.invoke({'notes':'blackhoel'})

print(result)
