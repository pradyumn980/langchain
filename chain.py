from langchain_core.prompts import PromptTemplate
# pyrefly: ignore [missing-import]
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
)

import os
# Ensure the key from Agents\.env is used, though it might be easier to just read it
with open(r"c:\Users\Pradyumn Agrahari\Desktop\Machine Learning\Agents\.env") as f:
    for line in f:
        if line.startswith("GEMINI_API_KEY="):
            os.environ["GOOGLE_API_KEY"] = line.strip().split("=")[1]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)

parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"topic": "cats"})

print(result)