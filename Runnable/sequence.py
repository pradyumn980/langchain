from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template='tell me a joke about {topic}',
    input_variables=['topic'],
)

prompt2=PromptTemplate(
    template='explain the joke {joke}',
    input_variables=['joke'],
)
parser=StrOutputParser()

chain=RunnableSequence(prompt,model,parser,prompt2|model|parser)

result=chain.invoke({'topic':'teachers'})

print(result)