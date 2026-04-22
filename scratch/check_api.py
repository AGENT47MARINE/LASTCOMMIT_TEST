
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

def check_api():
    models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
    for model_name in models:
        try:
            llm = ChatGroq(model=model_name, temperature=0)
            response = llm.invoke("Hi")
            print(f"Model {model_name}: SUCCESS")
        except Exception as e:
            print(f"Model {model_name}: FAILED - {str(e)[:100]}...")

if __name__ == "__main__":
    check_api()
