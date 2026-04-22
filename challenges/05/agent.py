from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# We use the recommended format from prompt_and_test.md
SYSTEM_PROMPT = """\
You are a precise answer engine. Determine if a given number is even, odd, or prime, or answer other direct boolean questions.

RULES:
1. Return ONLY "Yes." or "No." — nothing more, nothing less.
2. You MUST include the period at the end of "Yes." or "No." if the examples show it.
3. Never use conversational filler: no "Sure!", "Of course!", "Certainly!", "I think".
4. Never repeat or rephrase the question in your answer.
5. Do not add explanations, caveats, or extra commentary.
6. Output plain text only. No markdown, no bullet points.
"""

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

def run(query: str) -> str:
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Is 17 a prime number?"),
        ("ai", "Yes."),
        ("human", "Is 8 an odd number?"),
        ("ai", "No."),
        ("human", "Is 9 an odd number?"),
        ("ai", "Yes."),
        ("human", "Is 22 an even number?"),
        ("ai", "Yes."),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": query})
    return response.content.strip()
