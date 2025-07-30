import os
import getpass
import sys
import time
from langchain.chains.summarize.refine_prompts import prompt_template
from typing import Annotated, TypedDict, List, Optional
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from openai import max_retries
from langchain.agents import create_react_agent, AgentExecutor
from langchain.vectorstores import FAISS
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
import pandas as pd

"""
환경변수 설정
"""
load_dotenv(override=True)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")

print(f"# OPENAI_API_KEY : {os.environ.get("OPENAI_API_KEY")[:10]}")


"""
인트로 화면 정의
"""
ASCII_LOGO = r"""
░███     ░███   ░██████   ░██    ░██ ░██████░██████████                
░████   ░████  ░██   ░██  ░██    ░██   ░██  ░██                        
░██░██ ░██░██ ░██     ░██ ░██    ░██   ░██  ░██                        
░██ ░████ ░██ ░██     ░██ ░██    ░██   ░██  ░█████████                 
░██  ░██  ░██ ░██     ░██  ░██  ░██    ░██  ░██                        
░██       ░██  ░██   ░██    ░██░██     ░██  ░██                        
░██       ░██   ░██████      ░███    ░██████░██████████                
                                                                       
                                                                       
                                                                       
░█████████  ░██████████   ░██████    ░██████   ░██     ░██   ░██████   
░██     ░██ ░██          ░██   ░██  ░██   ░██   ░██   ░██   ░██   ░██  
░██     ░██ ░██         ░██        ░██           ░██ ░██   ░██         
░█████████  ░█████████  ░██         ░████████     ░████     ░████████  
░██   ░██   ░██         ░██                ░██     ░██             ░██ 
░██    ░██  ░██          ░██   ░██  ░██   ░██      ░██      ░██   ░██  
░██     ░██ ░██████████   ░██████    ░██████       ░██       ░██████   
                                                                       
                                                                       
"""

DESCRIPTION = """
🤖 SNZ Movie Recommendation Chatbot
-----------------------------------------
🔹 챗봇을 통해 오늘 하루를 털어놔보세요! 당신에게 맞는 영화를 추천해드립니다.
🔹 영화는 역시 최신보단 클래식이죠!
🔹 작성자 : 길나영, 김혜준, 송승호, 심재윤, 정준식, 최윤호

"""

def show_intro():
    os.system("clear" if os.name == "posix" else "cls")
    print(ASCII_LOGO)
    time.sleep(0.2)
    print(DESCRIPTION)
    time.sleep(0.5)



"""
변수 정의
"""

LLM_MODEL = "gpt-4o-mini"
DATA_FILE_PATH = "./imdb_top_1000.csv"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

"""
FAISS DB 구성
"""

df_movies = (
    pd.read_csv(DATA_FILE_PATH)
      .dropna(subset=["Series_Title", "Genre", "Overview"])
      .drop_duplicates(subset=["Series_Title"])
)

texts  = (
    df_movies["Series_Title"] + " | " +
    df_movies["Genre"]        + " | " +
    df_movies["Overview"]
).tolist()

metas = df_movies[["Series_Title", "Genre", "Overview"]].to_dict(orient="records")
vectorstore = FAISS.from_texts(texts, embedding_model, metadatas=metas)


"""
상태 정의
"""

class MyState(TypedDict):
    messages: List[dict]                 # 전체 대화 기록
    ended: bool                          # 종료 여부
    summary: List[str]                    # 영화 추천 요약
    recommendations: Optional[List[dict]] # 최종 추천 결과

"""
LLM 및 Embedding
"""
llm = ChatOpenAI(model=LLM_MODEL)

"""
Chatbot 노드 정의
"""
def supervisor_node(state: MyState) -> MyState:
    user_msgs = [m for m in state["messages"] if m["role"] == "user"]
    state["turn_count"] = len(user_msgs)
    return state

def route_decision(state: MyState) -> str:
    # 유저 메시지가 들어오면 챗봇으로
    if state["messages"] and state["messages"][-1]["role"] == "user":
        return "chatbot"
    # 챗봇이 응답했으면 영화 추천
    elif state["messages"] and state["messages"][-1]["role"] == "assistant":
        return "movie_rec"
    else:
        return "chatbot"
    
def chatbot_node(state: MyState) -> MyState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    messages = state["messages"]
    last_user_msg = messages[-1]["content"]

    # 친근한 답변
    convo = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
    prompt = f"""
    당신은 영화 추천을 위한 대화형 챗봇입니다.
    지금까지 대화:
    {convo}

    사용자의 마지막 발화: {last_user_msg}

    친근하게 한 문장으로 답변하세요.
    """
    bot_reply = llm.invoke(prompt).content.strip()
    messages.append({"role":"assistant","content":bot_reply})

    # 감정/취향 요약
    summary_prompt = f"""
    아래 대화에서 사용자의 감정 상태와 영화 취향만 1~2문장으로 요약하세요:
    {convo}
    """
    summary = llm.invoke(summary_prompt).content.strip()
    state["summary"] = [summary]

    state["messages"] = messages
    return state

def movie_rec_node(state: MyState) -> MyState:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    path = "imdb_top_1000.csv"
    df = pd.read_csv(path)
    df = df[['Series_Title','Genre','Overview']].dropna().drop_duplicates()

    summary = state["summary"][0]
    last_user_msg = state["messages"][-2]["content"]  # 마지막 유저 발화

    sample_df = df.sample(30, random_state=42)
    movies_text = "\n".join([
        f"{row.Series_Title} ({row.Genre}): {row.Overview}"
        for _, row in sample_df.iterrows()
    ])

    prompt = f"""
    사용자의 감정 및 취향 요약:
    {summary}

    사용자의 마지막 메시지:
    {last_user_msg}

    아래 후보 영화 목록에서 사용자의 취향에 맞는 5개를 선택하고
    추천 사유를 1~2문장씩 작성하세요.
    JSON 형식:
    [
        {{"title": "영화제목", "reason": "추천 사유"}},
        ...
    ]

    후보:
    {movies_text}
    """
    resp = llm.invoke(prompt).content

    import json
    try:
        recs = json.loads(resp)
    except:
        recs = [{"title":"추천 성공","reason":resp}]

    state["recommendations"] = recs
    print("=== 🎬 영화 추천 5선 ===")
    for r in recs:
        print(f"- {r['title']}: {r['reason']}")

    return state






def main():
    show_intro()
    
    # 앱 초기 UI
    workflow = StateGraph(MyState)

    # 노드 추가
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("chatbot", chatbot_node)
    workflow.add_node("movie_rec", movie_rec_node)

    # 시작 노드
    workflow.set_entry_point("supervisor")

    # supervisor → chatbot / movie_rec 조건부 이동
    workflow.add_conditional_edges(
        "supervisor",
        route_decision,
        {"chatbot": "chatbot", "movie_rec": "movie_rec"}
    )

    # chatbot → supervisor
    workflow.add_edge("chatbot", "supervisor")

    # movie_rec → end
    workflow.set_finish_point("movie_rec")

    # 최종 그래프 컴파일
    app = workflow.compile()
    
    # 유저 입력을 받아서 state 초기화
    user_input = input(":bust_in_silhouette: User: ")
    state = {
        "messages": [{"role": "user", "content": user_input}],
        "summary": [],
        "recommendations": []
    }
    # 워크플로우 실행
    state = app.invoke(state)
    # 챗봇 응답 출력
    print(":robot_face: 챗봇 응답:", state["messages"][-1]["content"])
    # 영화 추천 출력
    print(":clapper: 영화 추천:")
    for rec in state["recommendations"]:
        print(f"- {rec['title']}: {rec['reason']}")



if __name__ == "__main__":
    # 실행
    main() 