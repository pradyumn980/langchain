from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
import os
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough

load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.7,
)

model=ChatHuggingFace(llm=llm)

passthrough=RunnablePassthrough()

prompt=PromptTemplate(
    template='tell me a joke about {topic}',
    input_variables=['topic'],
)

prompt2=PromptTemplate(
    template='explain the joke {joke}',
    input_variables=['joke'],
)
parser=StrOutputParser()

joke_generate=RunnableSequence(prompt,model,parser)

parallel_chain=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2|model|parser)
})
final=RunnableSequence(joke_generate,parallel_chain)
result=final.invoke({'topic':'teachers'})

print(result)