import json
from langchain_community.tools import DuckDuckGoSearchResults

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
