from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    max_new_tokens=100,
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)
template1=PromptTemplate(
    template='write a detail report on {topic}',
    input_variables=['topic']
)

template2=PromptTemplate(
    template='write a 5 line summary on {text}',
    input_variables=['text']
)

prompt1=template1.invoke({'topic':'blackhole'})
result=model.invoke(prompt1)
print(result.content)

prompt2=template2.invoke({'text':result.content})
final=model.invoke(prompt2)


print(final.content)

