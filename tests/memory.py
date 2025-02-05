import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

load_dotenv()

API_KEY = os.getenv("API_KEY", None)


def get_chat_model():
    return ChatOpenAI(
        model="deepseek/deepseek-r1-distill-qwen-14b",
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        temperature=0.1,
    )


class ChatState:

    def __init__(self):
        self.messages = []


def chatbot_node(state):
    llm = get_chat_model()
    response = llm.invoke(state.messages)
    state.messages.append(AIMessage(content=response.content))
    return {"messages": state.messages}


def setup_workflow():
    workflow = StateGraph(dict)
    workflow.add_node("chatbot", chatbot_node)
    workflow.set_entry_point("chatbot")
    workflow.add_edge("chatbot", END)
    return workflow.compile()


def main():
    st.title("🤖 Memory Chatbot")

    # Initialize session state if not already set.
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "workflow" not in st.session_state:
        st.session_state.workflow = setup_workflow()

    # Display existing messages.
    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        st.chat_message(role).write(msg.content)
    # Only show the chat input when not loading.
    if not st.session_state.get("is_loading", False):
        prompt = st.chat_input("What's up?")
    else:
        prompt = None

    if prompt:
        # Display user's message immediately.
        user_message = HumanMessage(content=prompt)
        st.session_state.messages.append(user_message)
        st.rerun()  # Rerun to immediately show the user's prompt.

    if st.session_state.messages and (not isinstance(
            st.session_state.messages[-1], AIMessage)):
        current_state = ChatState()
        current_state.messages = st.session_state.messages.copy()

        # Set the loading flag so the chat input is hidden.
        st.session_state.is_loading = True

        with st.spinner("Generating response..."):
            st.chat_message("assistant").write("...")
            result = st.session_state.workflow.invoke(current_state)

        # Update the messages with the AI's response.
        st.session_state.messages = result["messages"]

        # Reset the loading flag when done.
        st.session_state.is_loading = False

        st.rerun()


if __name__ == "__main__":
    main()
