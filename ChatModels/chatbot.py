# pyrefly: ignore [missing-import]
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V4-Flash-0731",
    max_new_tokens=50,
    temperature=0.7,
)

model = ChatHuggingFace(llm=llm)
chat_history=[]
while True:
    user_input=input('You:')
    chat_history.append(user_input)
    if user_input=='exit':
        break
    result=model.invoke(chat_history)
    chat_history.append(result)
    print("AI:", result.content)

# result = model.invoke("What is the capital of India?")
# print(result.content)