from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda
from pydantic import BaseModel,Field
from typing import Literal

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)


parser=StrOutputParser()

class Feedback(BaseModel):
    sentiment:Literal['positive','negative']=Field(description='give the feedback of the sentiment')


parser2=PydanticOutputParser(pydantic_object=Feedback)

prompt1=PromptTemplate(
    template='classify wheter the feedback from user is positive or negative {response}  \n {format_instruction}',
    input_variables=['response'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain=prompt1|model|parser2


prompt2=PromptTemplate(
    template='Give a appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Give a appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)


branch_chain=RunnableBranch(
    (lambda x:x.sentiment=='positive',prompt2|model|parser),
    (lambda x:x.sentiment=='negative',prompt3|model|parser),
    RunnableLambda(lambda x:'could not find sentiment')
)

final_chain=classifier_chain|branch_chain

result=final_chain.invoke({'response':'this is good smartphone'})
print(result)
