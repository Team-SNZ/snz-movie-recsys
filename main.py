import os
import getpass
import matplotlib.pyplot as plt
from langchain.chains.summarize.refine_prompts import prompt_template
from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import max_retries



"""
환경변수 설정
"""
load_dotenv(override=True)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("TAVILY_API_KEY")

print(f"# OPENAI_API_KEY : {os.environ.get("OPENAI_API_KEY")[:10]}")
print(f"# TAVILY_API_KEY : {os.environ.get("TAVILY_API_KEY")[:10]}")


"""
변수 정의
"""
LLM_MODEL = "gpt-4o-mini"


"""
상태 정의
"""
class MyState(TypedDict):
    messages:Annotated[list, add_messages] 
    

"""
LLM 및 Tools 정의
"""
llm = ChatOpenAI(model=LLM_MODEL)
"""
Chatbot 노드 정의
"""
# 1. 챗봇 노드
def chatbot(state:MyState):
    if not state["message"]:
        current_message = "오늘 기분이 어때?"
    
    else:
        current_message = state["messages"] 
    response = llm.invoke(current_message)
    response_list = [response]
    return {"messages": [response]}

# 2. 감성 분석 노드
def emotion_analysis_node(state: MyState):
    full_history = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
    joined_text = "\n".join(full_history[-3:])
    prompt = f"""
    
    """


def main():
    """
    LangGraph 상태 그래프 정의
    """
    builder = StateGraph(MyState)
    builder.add_node("chatbot", chatbot)
    # builder.add_node("tools", tool_node)

    builder.add_edge(START, "chatbot")
    builder.add_edge("chatbot", END)

    graph = builder.compile()
    print(graph)

if __name__ == "__main__":
    main()