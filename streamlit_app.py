import os
import re
from typing import List, Tuple

import streamlit as st
from langchain_groq import ChatGroq


HANDBOOK_CHUNKS = [
    "Leave Policy: Employees are entitled to 18 paid leave days per calendar year, including casual and sick leave combined.",
    "Laptop Policy: Company laptops are provided to all full-time employees and must be returned upon exit. Personal use is permitted within reasonable limits.",
    "Remote Work Policy: Employees may work remotely up to 3 days per week with manager approval, submitted via the HR portal.",
    "Expense Policy: Business expenses including client meals and travel are reimbursable with receipts submitted within 30 days.",
    "Probation Policy: New employees undergo a 6-month probation period, reviewed at the 3-month and 6-month marks.",
    "Notice Period: Employees must serve a notice period of 60 days upon resignation, unless otherwise agreed with HR.",
    "Health Insurance: All employees are covered under group health insurance from day one, extending to immediate family.",
    "Working Hours: Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday, with flexible start times within a 1-hour window.",
    "Grievance Redressal: Employees can raise workplace grievances confidentially through the HR helpline, acknowledged within 2 working days.",
    "Exit Process: Employees must complete a knowledge transfer plan before their last working day; full settlement is processed within 45 days.",
]


def get_api_key() -> str:
    # Streamlit Cloud secrets are available in st.secrets; local runs can use env vars.
    if "GROQ_API_KEY" in st.secrets:
        return st.secrets["GROQ_API_KEY"]
    return os.getenv("GROQ_API_KEY", "")


@st.cache_resource
def get_llm() -> ChatGroq:
    api_key = get_api_key()
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY")
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=api_key)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def retrieve(question: str, k: int = 2) -> List[str]:
    # Lightweight lexical retrieval to keep deployment simple and fast.
    q_tokens = set(tokenize(question))
    scored: List[Tuple[int, str]] = []
    for chunk in HANDBOOK_CHUNKS:
        c_tokens = set(tokenize(chunk))
        overlap = len(q_tokens.intersection(c_tokens))
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for score, chunk in scored[:k] if score > 0]


def classify_question(llm: ChatGroq, question: str) -> str:
    prompt = f"""Classify the user's question as exactly one word: either policy or general.
policy = questions about company rules, leave, expenses, equipment, benefits, conduct, or HR processes.
general = anything else (small talk, general knowledge, unrelated topics).
Question: {question}
Answer:"""
    response = llm.invoke(prompt).content.strip().lower()
    return "policy" if "policy" in response else "general"


def answer_policy(llm: ChatGroq, question: str) -> Tuple[str, str]:
    chunks = retrieve(question, k=2)
    context = "\n".join(chunks) if chunks else "No relevant handbook context found."
    prompt = f"""Answer the question using ONLY the context below. Be concise.
Context:
{context}
Question: {question}
Answer:"""
    answer = llm.invoke(prompt).content
    return context, answer


def answer_general(llm: ChatGroq, question: str) -> Tuple[str, str]:
    answer = llm.invoke(question).content
    return "(no retrieval -- general question)", answer


def main() -> None:
    st.set_page_config(page_title="Handbook Assistant", page_icon="📘", layout="centered")
    st.title("📘 Handbook Assistant")
    st.caption("Policy questions are answered from handbook context. General questions use direct LLM response.")

    if not get_api_key():
        st.error("GROQ_API_KEY is missing.")
        st.info(
            "For Streamlit Cloud: open App Settings -> Secrets and add GROQ_API_KEY. "
            "For local run: set GROQ_API_KEY in your shell environment."
        )
        st.stop()

    llm = get_llm()

    question = st.text_input("Ask a question", placeholder="How many paid leave days do I get?")
    ask = st.button("Ask")

    if ask:
        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Thinking..."):
            category = classify_question(llm, question)
            if category == "policy":
                context, answer = answer_policy(llm, question)
            else:
                context, answer = answer_general(llm, question)

        st.subheader("Result")
        st.write("Category:", category)
        st.write("Answer:")
        st.success(answer)

        with st.expander("Context used"):
            st.write(context)


if __name__ == "__main__":
    main()
