# %%
from langchain_core.tools import tool
from PIL import Image
import requests
import os
from slack_sdk import WebClient
# %%
@tool
def search_movie_image_mcp(title: str) -> str:
    """
    영화 제목을 기반으로 포스터 이미지 URL을 검색합니다.
    """
    api_key = os.getenv("GOOGLE_CSE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    query = f"{title} movie poster site:imdb.com OR site:wikipedia.org"
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&searchType=image&num=1&key={api_key}&cx={cse_id}"

    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    return data["items"][0]["link"]
# %% [markdown]
# 이미지 URL을 받아서 로컬 파일로 저장하는 유틸 함수

# %%
def download_image_to_file(url: str, save_path: str = "poster.jpg") -> str:
    """
    이미지 URL을 받아 파일로 저장합니다.
    Args:
        url: 이미지의 전체 링크
        save_path: 저장할 파일 경로 (기본값: poster.jpg)
    Returns:
        저장된 파일 경로
    """
    res = requests.get(url)
    res.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(res.content)
    return save_path
