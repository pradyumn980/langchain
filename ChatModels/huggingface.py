# pyrefly: ignore [missing-import]
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    max_new_tokens=100,
    temperature=1,
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("tell me a joke in english")
print(result.content)