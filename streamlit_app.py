import os
import re
import html
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


def init_session_state() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def render_custom_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --brand-ink: #0f172a;
            --brand-deep: #0b3b2e;
            --brand-teal: #0f766e;
            --brand-blue: #2563eb;
            --panel: #ffffff;
            --line: #d8e4f0;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 0%, #eaf4ff 0%, transparent 38%),
                radial-gradient(circle at 95% 10%, #e8fff4 0%, transparent 40%),
                linear-gradient(180deg, #f7fbff 0%, #fcfff9 100%);
            color: var(--brand-ink);
        }

        .stTextInput input {
            color: var(--brand-ink) !important;
            background: #ffffff !important;
            border: 1px solid #cbd5e1 !important;
        }

        .stTextInput input::placeholder {
            color: #6b7280 !important;
        }

        .stTextInput input:focus {
            border: 1px solid #2563eb !important;
            box-shadow: 0 0 0 1px #2563eb !important;
        }

        .app-hero {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1.15rem 1.2rem;
            background: var(--panel);
            margin-bottom: 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        }

        .hero-title {
            font-family: 'Trebuchet MS', 'Segoe UI', sans-serif;
            font-weight: 700;
            font-size: 2rem;
            letter-spacing: 0.01em;
            color: var(--brand-deep);
            margin: 0;
        }

        .hero-text {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            color: #1f2937;
            margin-top: 0.4rem;
        }

        .small-note {
            font-family: 'Segoe UI', Tahoma, sans-serif;
            color: #374151;
            font-size: 0.9rem;
        }

        .status-chip {
            display: inline-block;
            padding: 0.4rem 0.65rem;
            border-radius: 999px;
            border: 1px solid #bfdbfe;
            background: #eff6ff;
            color: #1e3a8a;
            font-weight: 600;
            font-size: 0.88rem;
            margin-right: 0.4rem;
        }

        .stButton > button,
        [data-testid="stFormSubmitButton"] > button {
            font-weight: 700;
            border-radius: 10px !important;
            min-height: 2.8rem;
        }

        .stButton > button {
            background: var(--brand-teal) !important;
            color: #ffffff !important;
            border: 1px solid #0f5f59 !important;
        }

        .stButton > button p,
        [data-testid="stFormSubmitButton"] > button p {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            background: #0d5d57 !important;
            color: #ffffff !important;
        }

        [data-testid="stFormSubmitButton"] > button {
            background: var(--brand-blue) !important;
            border: 1px solid #1d4ed8 !important;
            color: #ffffff !important;
        }

        [data-testid="stFormSubmitButton"] > button:hover {
            background: #1e4fd1 !important;
        }

        .stButton > button:focus {
            color: #ffffff !important;
            outline: 2px solid #2563eb !important;
            outline-offset: 1px !important;
        }

        .qa-card {
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
            padding: 0.85rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }

        .qa-label {
            font-weight: 700;
            color: #1e3a8a !important;
            margin-bottom: 0.3rem;
        }

        .qa-answer {
            color: #111827 !important;
            line-height: 1.55;
            white-space: pre-wrap;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_status_bar(has_key: bool, llm_ready: bool) -> None:
    c1, c2, c3, c4 = st.columns([1.6, 1.6, 1.4, 1.1])
    with c1:
        st.markdown(
            f"<span class='status-chip'>Secret: {'Loaded' if has_key else 'Missing'}</span>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<span class='status-chip'>LLM: {'Ready' if llm_ready else 'Not ready'}</span>",
            unsafe_allow_html=True,
        )
    with c3:
        if st.button("Run health check", use_container_width=True):
            if not llm_ready:
                st.session_state["health_status"] = ("error", "LLM is not initialized.")
            else:
                try:
                    pong = get_llm().invoke("Reply with exactly: OK").content.strip()
                    st.session_state["health_status"] = ("success", f"Model reachable: {pong}")
                except Exception as exc:
                    st.session_state["health_status"] = ("error", f"Health check failed: {exc}")
    with c4:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    status = st.session_state.get("health_status")
    if status:
        level, message = status
        if level == "success":
            st.success(message)
        else:
            st.error(message)


def main() -> None:
    st.set_page_config(
        page_title="Handbook Assistant",
        page_icon="📘",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    render_custom_styles()
    init_session_state()

    st.markdown(
        """
        <section class="app-hero">
            <h1 class="hero-title">📘 Handbook Assistant</h1>
            <p class="hero-text">Ask policy questions to get handbook-grounded answers, or ask general questions for direct LLM help.</p>
            <p class="small-note">Routing mode: policy -> context answer, general -> direct answer</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        st.write("Try one:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("How many paid leave days do I get?", use_container_width=True):
                st.session_state["question_input"] = "How many paid leave days do I get?"
                st.rerun()
        with col2:
            if st.button("Tell me a fun fact about octopuses", use_container_width=True):
                st.session_state["question_input"] = "Tell me a fun fact about octopuses"
                st.rerun()

    has_key = bool(get_api_key())
    if not has_key:
        render_status_bar(has_key=False, llm_ready=False)
        st.error("GROQ_API_KEY is missing.")
        st.info(
            "For Streamlit Cloud: open App Settings -> Secrets and add GROQ_API_KEY. "
            "For local run: set GROQ_API_KEY in your shell environment."
        )
        st.stop()

    try:
        llm = get_llm()
        llm_ready = True
    except Exception as exc:
        render_status_bar(has_key=True, llm_ready=False)
        st.error(f"Failed to initialize LLM: {exc}")
        st.stop()

    render_status_bar(has_key=has_key, llm_ready=llm_ready)

    for i, item in enumerate(st.session_state.chat_history, start=1):
        q = html.escape(item["question"])
        c = html.escape(item["category"])
        a = html.escape(item["answer"])
        st.markdown(
            f"""
            <section class="qa-card">
                <div class="qa-label">Q{i}: {q}</div>
                <div><strong>Category:</strong> {c}</div>
                <div class="qa-answer">{a}</div>
            </section>
            """,
            unsafe_allow_html=True,
        )
        with st.expander(f"Context used for Q{i}"):
            st.write(item["context"])

    st.subheader("Ask a question")
    with st.form("ask_form", clear_on_submit=True):
        user_question = st.text_input(
            "Type your question",
            key="question_input",
            placeholder="How many paid leave days do I get?",
        ).strip()
        ask = st.form_submit_button("Ask question", type="primary", use_container_width=True)

    if ask:
        if not user_question:
            st.warning("Please type a question before asking.")
            st.stop()

        with st.spinner("Thinking..."):
            category = classify_question(llm, user_question)
            if category == "policy":
                context, answer = answer_policy(llm, user_question)
            else:
                context, answer = answer_general(llm, user_question)

        st.session_state.chat_history.append(
            {
                "question": user_question,
                "category": category,
                "answer": answer,
                "context": context,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
