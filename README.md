# 🤖 LangGraph Chatbot with DuckDuckGo Search Integration 🌐
This project demonstrates how to build a **conversational chatbot** using LangGraph, LangChain, and OpenAI's Mistral model. The chatbot is capable of interacting with users and performing web searches using DuckDuckGo. The state of the conversation is managed using a memory checkpointing system, allowing the chatbot to remember context across interactions. 🧠✨

## 🚀 Features

- 🤖 Conversational AI: The chatbot uses OpenAI's Mistral model to generate human-like responses.

- 🔍 Web Search Integration: The chatbot can perform web searches using DuckDuckGo to provide up-to-date information.

- 📝 State Management: The conversation state is managed using LangGraph's memory checkpointing system, allowing the chatbot to remember context across interactions.

- 🛠️ Tool Integration: The chatbot can invoke external tools (like DuckDuckGo Search) based on the conversation flow.

## 📋 Requirements
- Python 3.11+
- **langchain_community**
- **langchain_openai**
- **langgraph**

## 🧠 How It Works

1. State Management:
The chatbot uses a State class to manage the conversation history. Each message is stored in a list, and the state is preserved using LangGraph's MemorySaver.

2. Chatbot Node:
The chatbot node generates responses using the Mistral model. It can also invoke tools (like DuckDuckGo Search) when needed.

3. Tool Node:
The tools node handles external tool invocations, such as performing a web search.

4. Graph Construction:
The chatbot and tool nodes are connected in a graph structure using LangGraph. The graph determines the flow of the conversation and when to invoke tools.

5. Memory Checkpointing:
The conversation state is saved in memory, allowing the chatbot to resume interactions seamlessly.

## 🔧 Installation 

1. Clone the repository:
   ```bash
   git clone https://github.com/kabir0st/simple-chatbot-with-memory/
    ```

2. Install Dependencies:
   ```bash
   cd simple-chatbot-with-memory
   pip install -r requirements.txt
    ```

3. Run the app:
   ```bash
   python app.py
    ```

