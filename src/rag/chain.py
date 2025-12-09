# src/rag/chain.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough

PROMPT = """You are an intelligent assistant for a university information system. Your role is to provide accurate, helpful, and well-structured answers based on the provided context.

**Instructions:**
1. Carefully analyze the context and the question
2. Provide a clear, direct answer using ONLY information from the context
3. Structure your response with proper formatting (use bullet points, lists, or tables when appropriate)
4. If the context contains partial information, provide what's available and clearly state what's missing
5. If the answer is not available in the context, politely state: "I don't have information about this in the available documents."
6. For numerical data (fees, dates, etc.), present them clearly and accurately
7. Use professional, friendly language appropriate for a university setting

**Context:**
{context}

**Question:**
{question}

**Answer:**"""


def build_chain(llm_model, google_api_key=None, temperature=0.3):
    prompt = ChatPromptTemplate.from_template(PROMPT)

    llm = ChatGoogleGenerativeAI(
        model=llm_model,
        google_api_key=google_api_key,
        temperature=temperature
    )

    chain = (
        {
            "context": RunnablePassthrough(),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return chain

def ask(chain, context, question):
    return chain.invoke({"context": context, "question": question})
