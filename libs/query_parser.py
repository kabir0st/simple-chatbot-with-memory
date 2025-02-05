import re
import os
from langchain_community.chat_models import ChatOpenAI

API_KEY = os.getenv("API_KEY", None)


def query_parser(prompt):
    print('Improving User query . . .')
    book_prompt = """Act as a Prompt Enhancer AI that takes user-input prompts
     and transforms them into collection of keywords for semantic search.
    User Prompt: {prompt}

    All keywords must be enclosed in double asterisks.
    """
    model = ChatOpenAI(model="deepseek/deepseek-r1-distill-qwen-14b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key=API_KEY,
                       temperature=0.1)
    formatted_prompt = book_prompt.format(prompt=prompt)
    response = model.invoke(formatted_prompt)
    pattern = r'\*\*([\w\s]+)\*\*'
    keywords = re.findall(pattern, response.content)
    if keywords:
        return " ,".join(keywords)
    return response.content
