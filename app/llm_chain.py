"""
llm_chain.py
Sets up a Gemini 3.1 Flash-Lite chat model via LangChain, grounded with a
small curated fact-sheet for the vegetable the CNN just identified, so
answers stay relevant and are less likely to hallucinate.
"""

import json
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

FACTS_PATH = os.path.join("app", "vegetable_facts.json")


@st.cache_resource(show_spinner=False)
def load_facts() -> dict:
    with open(FACTS_PATH, "r") as f:
        return json.load(f)


def get_llm():
    """
    Instantiate the Gemini 3.1 Flash-Lite chat model.
    Reads the API key from Streamlit secrets (st.secrets["GOOGLE_API_KEY"])
    so it works both locally (.streamlit/secrets.toml) and on
    Streamlit Community Cloud (App settings -> Secrets).
    """
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY not found. Add it to .streamlit/secrets.toml locally, "
            "or to your Streamlit Community Cloud app's Secrets settings."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite",
        google_api_key=api_key,
        temperature=0.4,
        max_output_tokens=400,
    )


def build_system_prompt(vegetable_name: str) -> str:
    facts = load_facts()
    fact_snippet = facts.get(
        vegetable_name,
        "No curated fact-sheet is available for this item yet; "
        "answer from general knowledge but say so if unsure.",
    )
    return (
        "You are a friendly vegetable expert embedded in a mobile scanning app. "
        f"The user just scanned a photo and the model identified it as: {vegetable_name}. "
        f"Here are some verified reference facts about it:\n{fact_snippet}\n\n"
        "Answer the user's questions about THIS specific vegetable — nutrition, "
        "storage, selection tips, cooking ideas, substitutions, etc. "
        "Prefer the reference facts above when relevant, and clearly say when "
        "you're going beyond them with general knowledge. "
        "If asked something unrelated to this vegetable or food in general, "
        "gently steer the conversation back. Keep answers concise and practical."
    )


def get_response(vegetable_name: str, chat_history: list, user_message: str) -> str:
    """
    chat_history: list of {"role": "user"/"assistant", "content": str}
    Returns the assistant's reply as a plain string.
    """
    llm = get_llm()
    messages = [SystemMessage(content=build_system_prompt(vegetable_name))]

    for turn in chat_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))

    messages.append(HumanMessage(content=user_message))

    response = llm.invoke(messages)
    return response.content[0]["text"]
