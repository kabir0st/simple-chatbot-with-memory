from langchain_core.chat_history import (BaseChatMessageHistory,
                                         InMemoryChatMessageHistory)

STORE = {}


def get_purchase_history(user_id):
    return {}


# this is for demo
def get_session_history(user_id: str) -> BaseChatMessageHistory:
    if user_id not in STORE:
        STORE[user_id] = InMemoryChatMessageHistory()
    return STORE[user_id]
