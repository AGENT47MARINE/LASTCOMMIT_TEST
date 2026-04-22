from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# We use the recommended format from prompt_and_test.md
SYSTEM_PROMPT = """\
You are a precise answer engine. Determine the highest, lowest, fastest, or otherwise requested value from a given text comparison.

RULES:
1. Return ONLY the extracted name or value verbatim — nothing more, nothing less.
2. Never use conversational filler: no "Sure!", "Of course!", "Certainly!", "I think", "Here is", "Great question", or any similar phrases.
3. Never repeat or rephrase the question in your answer.
4. Do not add explanations, caveats, or extra commentary unless explicitly asked.
5. Do not include any punctuation at the end of your answer.
6. Omit leading articles ("The", "A", "An") from the extracted entity. If the text says "The red balloon", you must output "red balloon".
7. Output plain text only. No markdown, no bullet points, no headers.
"""

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def run(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Alice scored 80, Bob scored 90. Who scored highest?"),
        ("ai", "Bob"),
        ("human", "Charlie ran 10km, David ran 12km. Who ran more?"),
        ("ai", "David"),
        ("human", "The red balloon reached 50ft, the blue balloon reached 40ft. Which went higher?"),
        ("ai", "red balloon"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return response.content.strip()
