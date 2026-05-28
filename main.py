from dotenv import load_dotenv
import os
import langchain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from openai import base_url

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")


llm = ChatOpenAI(
    model_name="openai/gpt-oss-20b:free", 
    temperature=0,
    api_key= api_key,
    base_url="https://openrouter.ai/api/v1/"
)

def main():
    question = input("Enter your question: ")
    message = HumanMessage(content=question)
    response = llm.invoke([message])
    print("Response:", response.content)


if __name__ == "__main__":
    main()
