import os
import getpass
import sys
import time
import json  # 추가 필요
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

load_dotenv(override=True)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
print(f"# OPENAI_API_KEY : {os.environ.get('OPENAI_API_KEY')[:10]}")

LLM_MODEL          = "gpt-4o-mini"
EMBED_MODEL        = "text-embedding-3-small"
DATA_FILE_PATH     = "./snz_movie_recsys/imdb_top_1000.csv"
TOP_K              = 30                
BOT_TEMP           = 0                 
TURN_LIMIT         = 3  # 3번의 대화로 변경

class MyState(TypedDict):
    messages      : List[dict]                 
    ended         : bool                       
    summary       : List[str]                  
    candidates    : Optional[List[dict]]       
    recommendations: Optional[List[dict]]
    turn_count    : int  # 턴 카운트 추가

# ---------- LLM & 임베딩 ----------
llm_chat = ChatOpenAI(model=LLM_MODEL, temperature=BOT_TEMP)
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL)

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

def chatbot(state: MyState) -> MyState:
    msgs = state["messages"]
    turn_count = state.get("turn_count", 0)
    
    # 사용자 메시지만 카운트 (role이 "user"인 메시지)
    user_messages = [m for m in msgs if m["role"] == "user"]
    current_turn = len(user_messages)
    
    if current_turn == 0:
        # 첫 번째 인사
        reply = "안녕하세요! 오늘 기분이 어떠신가요? 어떤 영화를 보고 싶으신지 알려주세요."
    else:
        last_user = msgs[-1]["content"] if msgs else ""
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        
        if current_turn < TURN_LIMIT:
            prompt = f"""당신은 친근한 영화 추천 챗봇입니다.
지금까지 대화:
{convo}

사용자 마지막 발화: {last_user}
현재 {current_turn}번째 대화입니다. {TURN_LIMIT}번의 대화 후 영화를 추천할 예정입니다.
한 문장으로 답하고, 영화를 추천하기 위한 정보를 자연스럽게 더 물어보세요."""
        else:
            prompt = f"""당신은 친근한 영화 추천 챗봇입니다.
지금까지 대화:
{convo}

{TURN_LIMIT}번의 대화가 완료되었습니다. 이제 영화를 추천해드리겠습니다!
간단한 마무리 인사와 함께 곧 추천을 시작한다고 말해주세요."""
        
        reply = llm_chat.invoke(prompt).content.strip()
    
    msgs.append({"role": "assistant", "content": reply})
    
    # 3번의 사용자 입력 후 종료
    user_turn_count = len([m for m in msgs if m["role"] == "user"])
    ended = user_turn_count >= TURN_LIMIT
    
    if ended:
        # 요약 생성
        convo = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
        summary_prompt = f"""다음 대화를 1~2문장으로 영화 취향·감정을 요약:
{convo}"""
        summary = llm_chat.invoke(summary_prompt).content.strip()
        state["summary"] = [summary]
        print(f"\n💭 대화 요약: {summary}")

    state.update({
        "messages": msgs, 
        "ended": ended,
        "turn_count": turn_count + 1
    })
    return state

# ---------- 2) Retrieval 노드 ----------
def retrieve_movies(state: MyState) -> MyState:
    """요약된 대화를 바탕으로 추천 영화를 찾음."""
    if not state.get("summary"):
        return state 

    query = state["summary"][0]
    docs = vectorstore.similarity_search(query, k=TOP_K)

    candidates = [
        {
            "title": d.metadata["Series_Title"],
            "genre": d.metadata["Genre"],
            "overview": d.metadata["Overview"],
        }
        for d in docs
    ]
    state["candidates"] = candidates
    print(f"\n🔍 {len(candidates)}개의 후보 영화를 찾았습니다.")
    return state

# ---------- 3) Recommendation 노드 ----------
def movie_recommendation(state: MyState) -> MyState:
    """Retriever 결과 중 5편의 영화와 각각의 추천 사유 생성"""
    if not state.get("candidates"):
        return state

    candidates_text = "\n".join(
        f"{c['title']} ({c['genre']}): {c['overview']}" for c in state["candidates"]
    )

    prompt = f"""사용자 영화 취향 요약:
{state['summary'][0]}

아래 후보 영화 중 5편을 골라 1~2문장 추천 사유와 함께 JSON 배열 형식으로 답하세요.
{candidates_text}

예시 형태:
[
  {{"title": "영화제목", "reason": "추천 사유"}},
  ...
]"""
    resp = llm_chat.invoke(prompt).content.strip()
    try:
        recs = json.loads(resp)
    except json.JSONDecodeError:
        recs = [{"title": "추천 성공!!!", "reason": resp}]

    state["recommendations"] = recs
    return state

# ---------- 조건부 엣지를 위한 함수 ----------
def should_continue(state: MyState) -> str:
    """대화 계속 여부를 결정하는 함수"""
    if state["ended"]:
        return "retrieve"
    else:
        return "chatbot"

# ---------- LangGraph 정의 ----------
builder = StateGraph(MyState)

# 노드 추가
builder.add_node("chatbot", chatbot)
builder.add_node("retrieve", retrieve_movies)
builder.add_node("recommend", movie_recommendation)

# 시작 → chatbot
builder.add_edge(START, "chatbot")

# chatbot 실행 뒤 조건부 분기
builder.add_conditional_edges(
    "chatbot",
    should_continue,  # 조건 함수
    {
        "chatbot": "chatbot",   # ended == False → 다시 chatbot (루프)
        "retrieve": "retrieve"  # ended == True  → 검색 단계
    }
)

# 검색 → 추천 → END
builder.add_edge("retrieve", "recommend")
builder.add_edge("recommend", END)

graph = builder.compile()

# ---------- 멀티턴 대화 실행 함수 ----------
def run_conversation():
    """멀티턴 대화를 진행하는 함수"""
    state: MyState = {
        "messages": [],
        "ended": False, 
        "summary": [], 
        "candidates": None, 
        "recommendations": None,
        "turn_count": 0
    }
    
    print("🎬 영화 추천 챗봇에 오신 것을 환영합니다!")
    print("3번의 대화를 통해 당신에게 맞는 영화를 추천해드리겠습니다.\n")
    
    # 첫 번째 챗봇 응답 (인사)
    state = chatbot(state)
    print(f"🤖: {state['messages'][-1]['content']}")
    
    # 3번의 사용자 입력을 받음
    for turn in range(TURN_LIMIT):
        user_input = input(f"\n👤 ({turn+1}/{TURN_LIMIT}): ")
        
        # 사용자 메시지 추가
        state["messages"].append({"role": "user", "content": user_input})
        
        # 챗봇 응답 생성
        state = chatbot(state)
        
        print(f"🤖: {state['messages'][-1]['content']}")
    
    print("\n" + "="*50)
    print("대화가 완료되었습니다! 영화를 추천해드리겠습니다...")
    print("="*50)
    
    # 검색 및 추천 단계 실행
    state = retrieve_movies(state)
    state = movie_recommendation(state)
    
    # 최종 추천 결과 출력
    print("\n🎬 최종 추천 영화")
    if state.get("recommendations"):
        for i, r in enumerate(state["recommendations"], 1):
            print(f"{i}. {r['title']}")
            print(f"   💡 {r['reason']}\n")
    else:
        print("추천 결과가 없습니다.")

# ---------- 실행 ----------
if __name__ == "__main__":
    run_conversation()