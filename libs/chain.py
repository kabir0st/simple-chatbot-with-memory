import re
from typing import Annotated, Sequence, TypedDict

from langchain.prompts import PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import AnyMessage, add_messages

from libs.query_parser import query_parser as llm_query_parser
from libs.reteriver import get_qdrant_retriever
from libs.users import get_purchase_history

RECOMMENDATION_TEMPLATE = """As a book recommendation assistant, follow
these rules:
1. Strict Context Adherence: Only recommend books explicitly listed in the
catalog context below. Never invent titles.
2. Exact Title Matching: Use the full, exact titles as they appear in the
context. No abbreviations or variations.
3. Anti-Duplication: Do not repeat books from the user's purchase history:
{purchase_history}

Recommend 5-10 books matching the user's interest: {question}

For each book, include:
- Title (EXACTLY as in the context)
- Author(s)
- Genres
- Reason why you recommend it

Keywords to guide recommendations: {parsed_query_keywords}

Catalog Context (ALL recommendations must come from here):
{context}

Final Validation:
- Double-check every title against the context.
- If a book isn't in the context, exclude it."""

PROMPT = PromptTemplate(template=RECOMMENDATION_TEMPLATE,
                        input_variables=[
                            "context", "question", "parsed_query_keywords",
                            "formatted_history"
                        ])


class State(TypedDict):
    user_id: str
    question: str
    chat_history: Annotated[Sequence[AnyMessage], add_messages]
    parsed_query: str
    purchase_history: list
    context: list
    recommendations: str


def parse_query(state: State):
    print('query parser triggered', state["question"])
    return {"parsed_query": llm_query_parser(state["question"])}


def retrieve_purchase_history(state: State):
    print('purchase history', state["user_id"])
    return {"purchase_history": get_purchase_history(state["user_id"])}


def format_chat_history(state: State):
    formatted = "\n".join(f"{msg.type}: {msg.content}"
                          for msg in state["chat_history"])
    print('chat history')
    return {"formatted_history": formatted}


def retrieve_context(state: State, q_client, embedding_model):
    print('context ')
    retriever = get_qdrant_retriever(q_client, embedding_model)
    return {"context": retriever.invoke(state["parsed_query"])}


def generate_recommendations(state: State, model):
    context_str = "\n\n".join(doc.page_content for doc in state["context"])

    prompt = PROMPT.format(
        context=context_str,
        question=state["question"],
        parsed_query_keywords=state["parsed_query"],
        purchase_history=", ".join(state["purchase_history"]),
        formatted_history=state.get("formatted_history", ""))
    print('recommendataion')
    return {"recommendations": model.invoke(prompt).content}


def validate_recommendations(state: State):
    print('validate')
    context_titles = set()
    for doc in state["context"]:
        match = re.search(r"Title: (.+)", doc.page_content)
        if match:
            context_titles.add(match.group(1).strip().lower())

    valid_output = []
    for line in state["recommendations"].split("\n"):
        if line.startswith("- Title: "):
            title = line.split(": ")[1].strip().lower()
            if title not in context_titles:
                continue
        valid_output.append(line)

    return {"recommendations": "\n".join(valid_output)}


def update_chat_history(state: State):
    print('updating chathistory')
    new_history = list(state["chat_history"])
    new_history.append(HumanMessage(content=state["question"]))
    new_history.append(AIMessage(content=state["recommendations"]))
    return {"chat_history": new_history}


def create_recommendation_graph(model, q_client, embedding_model):
    workflow = StateGraph(State)

    # Define nodes
    workflow.add_node("parse_query", parse_query)
    workflow.add_node("retrieve_purchase_history", retrieve_purchase_history)
    workflow.add_node("format_chat_history", format_chat_history)
    workflow.add_node(
        "retrieve_context",
        lambda state: retrieve_context(state, q_client, embedding_model))
    workflow.add_node("generate_recommendations",
                      lambda state: generate_recommendations(state, model))
    workflow.add_node("validate_recommendations", validate_recommendations)
    workflow.add_node("update_chat_history", update_chat_history)
    print('Defining Edges . . .')
    # Define edges
    workflow.set_entry_point("parse_query")
    workflow.add_edge("parse_query", "retrieve_purchase_history")
    workflow.add_edge("retrieve_purchase_history", "format_chat_history")
    workflow.add_edge("format_chat_history", "retrieve_context")
    workflow.add_edge("retrieve_context", "generate_recommendations")
    workflow.add_edge("generate_recommendations", "validate_recommendations")
    workflow.add_edge("validate_recommendations", "update_chat_history")
    workflow.add_edge("update_chat_history", END)

    return workflow
