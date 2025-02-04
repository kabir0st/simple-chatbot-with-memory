from typing import Annotated

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_community.tools import DuckDuckGoSearchResults

API_KEY = ("")


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

LLM = ChatOpenAI(model="mistralai/mistral-small-24b-instruct-2501",
                 base_url="https://openrouter.ai/api/v1",
                 api_key=API_KEY,
                 temperature=0.1)


def chatbot(state: State):
    return {"messages": [LLM.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

graph_builder.add_edge(START, "chatbot")

tool = DuckDuckGoSearchResults(max_results=5)

model_generated_tool_call = {
    "args": {
        "query": "euro 2024 host nation"
    },
    "id": "1",
    "name": "tavily",
    "type": "tool_call",
}

tool_msg = tool.invoke(model_generated_tool_call)

graph = graph_builder.compile()


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [{
            "role": "user",
            "content": user_input
        }]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except Exception:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
