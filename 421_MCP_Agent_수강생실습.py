# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 420. 수강생 실습 — MCP Agent (표준 프로토콜로 도구 연결하기)
#
# ## 학습 목표
# [400 Tools & Agents](400_Tools_Agents.py) 에서 `@tool` 데코레이터로 **같은 코드 안에**
# 도구를 정의했다면, 이번에는 **MCP (Model Context Protocol)** 라는 표준 규약으로
# **외부 서버가 제공하는 도구** 를 가져와 에이전트에 연결합니다.
#
# ## MCP 란?
# - Anthropic 이 2024년 11월 공개한 오픈 표준.
# - "AI 를 위한 **USB-C**" 라는 비유로 자주 설명됩니다.
# - LLM ↔ 외부 도구·데이터 사이의 연결 방식을 통일한 규약 — 한 번 만들어 두면
#   **MCP 를 지원하는 모든 모델·프레임워크가 재사용** 할 수 있습니다.
# - "도구를 만드는 사람" 과 "도구를 쓰는 사람" 이 깔끔하게 분리됩니다.
#
# ## 400 (`@tool`) vs 420 (MCP) 의 차이
# | 구분 | **400. `@tool` 도구** | **420. MCP 도구** |
# |------|----------------------|-------------------|
# | 정의 위치 | 에이전트 코드 안에 함께 | **별도 서버 프로세스** 가 제공 |
# | 재사용 | 해당 코드에서만 | **표준 프로토콜로 어디서나** |
# | 배포 | 코드를 같이 옮겨야 | 서버 URL 만 공유하면 됨 |
# | 비유 | 내 컴퓨터에 직접 짠 유틸 | USB-C 케이블 꽂듯 외부 도구 연결 |
# | 적합한 경우 | 프로젝트 전용 간단한 도구 | 공용·외부 제공 도구 |
#
# ## MCP 의 구성
# | 역할 | 설명 |
# |------|------|
# | **Client** | 모델·에이전트 쪽. MCP Server 에 접속해 도구를 가져옴 |
# | **Server** | 도구를 제공하는 쪽. Tools / Resources / Prompts 를 노출 |
#
# 이번 실습에서는 간단한 **수학 계산 MCP Server** 를 직접 만들고, LangChain 에이전트를
# Client 로 연결합니다.
#
# ## 참고 링크
# - [MCP 공식 문서](https://modelcontextprotocol.io/)
# - [LangChain 공식 문서 - MCP](https://docs.langchain.com/oss/python/langchain/mcp)

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.
#
# - `langchain`, `langchain-google-genai` : 에이전트 및 Gemini 모델 (400 노트북과 동일)
# - `langchain-mcp-adapters` : MCP Server 의 도구를 LangChain 도구로 변환
# - `mcp` : MCP Server 를 만들기 위한 공식 Python SDK (HTTP 서버 실행 기능 포함)

# %%
# !pip install -q -U langchain langchain-google-genai langchain-mcp-adapters mcp

# %% [markdown]
# ---
# ## 과제 1. 환경 변수 로드 + Gemini 모델 초기화
#
# 400 노트북과 동일하게 `.env` 의 `GOOGLE_API_KEY` 를 로드하고 Gemini 모델을 만듭니다.
# MCP 도구를 호출할 **Client (모델)** 쪽 준비입니다.
#
# **할 일**:
# - `load_dotenv()` 로 환경 변수를 로드하세요.
# - `init_chat_model("gemini-2.5-flash", ...)` 로 모델을 만드세요.

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %%
from langchain.chat_models import init_chat_model

# 모델 초기화 (400 노트북과 동일하게 Gemini 사용)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model

# %% [markdown]
# ---
# ## 과제 2. MCP Server 코드 작성
#
# MCP Server 는 도구를 제공하는 **별도의 프로그램** 입니다. 공식 SDK 의 `FastMCP` 를 쓰면
# `@mcp.tool()` 데코레이터만으로 손쉽게 만들 수 있습니다 (400 의 `@tool` 과 거의 동일).
#
# ### 연결 방식(transport) 두 가지
# | 방식 | 설명 |
# |------|------|
# | `stdio` | 데스크톱 앱(Claude Desktop, Cursor, VS Code 등) 이 로컬 서버를 하위 프로세스로 띄울 때 사용 |
# | `streamable_http` | 서버를 HTTP 로 띄워 두고 URL 로 접속 (원격 서버에 접속하는 방식) |
#
# Colab / Jupyter 노트북에서는 `stdio` 가 표준 입출력 제약(`fileno` 미지원)으로 동작하지 않습니다.
# 따라서 이 실습은 **`streamable_http`** 방식으로 서버를 띄우고 접속합니다.
#
# **할 일**:
# - `math_server_code` 문자열에 MCP Server 코드를 작성하고 `math_server.py` 로 저장하세요.
# - 서버는 `add(a, b)` 와 `multiply(a, b)` 두 도구를 노출해야 합니다.
#
# **힌트**:
# - 서버 측 transport 이름은 **하이픈** (`"streamable-http"`).
# - 클라이언트(`langchain-mcp-adapters`) 측 transport 이름은 **언더스코어** (`"streamable_http"`).
# - 이름이 다른 두 가지를 헷갈리지 않도록 주의하세요.

# %%
# MCP Server 코드를 문자열로 정의한 뒤 파일로 저장합니다.
math_server_code = '''
from mcp.server.fastmcp import FastMCP

# MCP Server 객체 생성 ("Math"는 서버 이름, host/port 는 HTTP 접속 주소)
mcp = FastMCP("Math", host="127.0.0.1", port=8000)


@mcp.tool()
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b


if __name__ == "__main__":
    # streamable-http 방식으로 서버 실행 (기본 주소: http://127.0.0.1:8000/mcp)
    # 주의: MCP SDK 의 transport 이름은 하이픈("streamable-http")입니다.
    #       클라이언트(langchain-mcp-adapters)는 언더스코어("streamable_http")를 쓰므로 혼동 주의.
    mcp.run(transport="streamable-http")
'''

with open("math_server.py", "w", encoding="utf-8") as f:
    f.write(math_server_code)

print("math_server.py MCP Server 파일을 생성했습니다.")

# %% [markdown]
# **관찰 포인트**
# - `@mcp.tool()` 데코레이터는 400 의 `@tool` 과 형태가 거의 같습니다 — type hint 와
#   docstring 만 있으면 도구의 입력 스키마와 설명이 자동 생성됩니다.
# - 이 파일 하나가 곧 **독립적으로 실행되는 MCP Server** 가 됩니다.

# %% [markdown]
# ---
# ## 과제 3. MCP Server 백그라운드 실행
#
# `streamable_http` 서버는 실행되면 계속 떠 있어야 하므로, 노트북 셀을 막지 않도록
# **별도의 백그라운드 프로세스** 로 띄웁니다. 서버 출력(로그·오류) 은 `mcp_server.log`
# 파일로 보냅니다 — 문제가 생기면 이 파일을 확인하세요.
#
# **할 일**:
# - `subprocess.Popen` 으로 `python math_server.py` 를 백그라운드 실행하세요.
# - `wait_for_server()` 함수로 포트가 열릴 때까지 기다리세요 (최대 30초).
# - 셀 재실행에 대비해 이전 `server_process` 가 있으면 먼저 종료하세요.
#
# **힌트**: 서버가 바로 응답하지 않아도 포트가 열리고 초기화에 잠시 시간이 필요할 수 있어
# `time.sleep(1)` 한 줄을 추가로 둡니다.

# %%
import subprocess
import sys
import socket
import time

# 이미 실행 중인 서버가 있으면 먼저 종료합니다 (셀 재실행 대비).
if "server_process" in globals():
    server_process.terminate()

# MCP Server 를 백그라운드 프로세스로 실행 (출력은 로그 파일로)
log_file = open("mcp_server.log", "w", encoding="utf-8")
server_process = subprocess.Popen(
    [sys.executable, "math_server.py"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
)


def wait_for_server(host="127.0.0.1", port=8000, timeout=30):
    """지정한 host:port 에 접속될 때까지 최대 timeout 초 동안 기다립니다."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


# 서버가 포트를 열 때까지 대기
if wait_for_server():
    time.sleep(1)  # 서버 내부 초기화를 위한 여유 시간
    print(f"MCP Server 실행됨 (PID: {server_process.pid})")
else:
    print("서버 기동 실패 — mcp_server.log 파일을 확인하세요.")

# %% [markdown]
# **관찰 포인트**
# - 같은 셀을 두 번 실행해도 안전합니다 — `if "server_process" in globals()` 가 이전
#   프로세스를 정리해 줍니다.
# - 서버가 시작되었지만 도구 호출이 실패한다면 `mcp_server.log` 의 stack trace 가 가장 먼저 봐야 할 단서입니다.

# %% [markdown]
# ---
# ## 과제 4. MCP Client 로 서버에 연결
#
# `MultiServerMCPClient` 는 **여러 MCP Server 에 동시에 연결** 할 수 있는 Client 입니다.
# 각 서버를 dict 로 등록하며, 본 실습에서는 방금 띄운 로컬 서버 1개만 등록합니다.
#
# **할 일**:
# - `MultiServerMCPClient({"math": {"url": ..., "transport": "streamable_http"}})` 형태로 client 를 만드세요.
#
# **힌트**:
# - `"math"` 는 우리가 임의로 정한 서버 별명 — 여러 서버를 등록할 때 구분용입니다.
# - 다른 MCP 서버 (예: 파일시스템, GitHub, Slack 등 공개된 MCP 서버) 를 추가하려면
#   같은 dict 에 항목을 더 넣으면 됩니다 — **확장이 매우 간단** 한 것이 MCP 의 강점입니다.

# %%
from langchain_mcp_adapters.client import MultiServerMCPClient

# 연결할 MCP Server 목록 정의
client = MultiServerMCPClient(
    {
        "math": {
            "url": "http://127.0.0.1:8000/mcp",   # 서버 접속 주소
            "transport": "streamable_http",       # 통신 방식
        },
        # 다른 서버를 추가하려면 여기에 항목을 더 넣으면 됩니다.
    }
)

# %% [markdown]
# ---
# ## 과제 5. MCP 도구 가져오기 (`get_tools`)
#
# `get_tools()` 는 등록된 모든 MCP Server 에 접속해, 서버가 제공하는 도구들을
# **LangChain 도구 객체로 변환** 해서 돌려줍니다.
#
# **할 일**:
# - `mcp_tools = await client.get_tools()` 로 도구 리스트를 가져오세요.
# - 도구의 개수와 이름·설명을 출력해 어떤 도구가 노출되어 있는지 확인하세요.
#
# **힌트**:
# - `get_tools()` 는 **비동기(async) 함수** 입니다 → `await` 로 호출.
# - Jupyter 노트북 셀은 top-level `await` 를 지원하므로 별도 이벤트 루프 설정이 필요 없습니다.

# %%
# MCP Server 의 도구를 LangChain 도구로 변환
mcp_tools = await client.get_tools()

print(f"가져온 도구 개수: {len(mcp_tools)}")
for t in mcp_tools:
    print(f"- {t.name}: {t.description}")

# %% [markdown]
# **관찰 포인트**
# - 가져온 도구 객체는 LangChain 의 일반 `Tool` 과 **인터페이스가 같습니다** —
#   `.name`, `.description`, `.invoke({...})` 모두 동일하게 동작합니다.
# - 즉 에이전트 입장에서는 이 도구가 MCP 출신인지 `@tool` 출신인지 알 필요가 없습니다.

# %% [markdown]
# ---
# ## 과제 6. MCP 도구로 에이전트 생성 및 실행
#
# 변환된 도구는 LangChain 의 일반 도구와 동일하므로, 400 노트북의 `create_agent` 를
# **그대로** 사용할 수 있습니다. MCP 도구든 `@tool` 도구든 에이전트 입장에서는 차이가 없습니다.
#
# **할 일**:
# - `create_agent(model, tools=mcp_tools, system_prompt=...)` 로 에이전트를 만드세요.
# - "12 더하기 8 을 한 다음, 그 결과에 5 를 곱하면?" 질문으로 `await agent.ainvoke(...)` 를 호출하세요.
#
# **힌트**: MCP 도구는 내부적으로 비동기로 동작하므로 노트북에서는 **비동기 API**
# (`ainvoke`, `astream`) 로 호출하는 것이 안전합니다.

# %%
from langchain.agents import create_agent

agent = create_agent(
    model,
    tools=mcp_tools,
    system_prompt="당신은 도움이 되는 어시스턴트입니다. 계산이 필요하면 주어진 도구를 사용하세요.",
)

print("MCP 도구로 에이전트가 생성되었습니다.")
print(f"사용 가능한 도구: {[t.name for t in mcp_tools]}")

# %%
# 에이전트 호출 (비동기)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "12 더하기 8을 한 다음, 그 결과에 5를 곱하면?"}]}
)

print("에이전트 응답:")
result["messages"][-1].pretty_print()

# %% [markdown]
# **관찰 포인트**
# - "더한 다음 곱하라" 는 복합 연산이므로 에이전트가 `add` → `multiply` 를 **순차 호출** 합니다.
# - 두 도구 모두 MCP Server 의 백그라운드 프로세스에서 실행되어 결과만 모델로 돌아옵니다 —
#   에이전트 코드에는 계산 로직이 한 줄도 없습니다.

# %% [markdown]
# ---
# ## 과제 7. 추론 과정 스트리밍 (`astream`)
#
# `astream` 으로 실행하면 에이전트가 어떤 MCP 도구를 어떤 인수로 호출하는지 **단계별로**
# 확인할 수 있습니다.
#
# **할 일**:
# - "7 과 6 을 곱한 값은?" 질문으로 `agent.astream(..., stream_mode="values")` 를 호출하세요.
# - 매 step 의 `event["messages"][-1].pretty_print()` 로 흐름을 시각화하세요.

# %%
async for event in agent.astream(
    {"messages": [{"role": "user", "content": "7과 6을 곱한 값은?"}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

# %% [markdown]
# **관찰 포인트 — MCP 도구 호출의 ReAct 흐름**
# - System → Human(질문) → AI 가 `multiply(a=7, b=6)` 도구 호출 결정 → Tool 결과(42) →
#   AI 최종 답변. 410 의 ReAct 흐름과 **정확히 같은 패턴** 입니다.
# - 다른 점은 도구 실행이 **외부 MCP Server 프로세스** 에서 일어난다는 것뿐 — 에이전트
#   입장에서는 검은 상자입니다.

# %% [markdown]
# ---
# ## 과제 8. MCP 도구 + 로컬 도구 함께 쓰기
#
# 실전에서는 외부 표준 도구(MCP) 와 내가 만든 전용 도구(`@tool`) 를 **섞어서** 하나의
# 에이전트에 넘기는 경우가 많습니다.
#
# **할 일**:
# - `@tool` 데코레이터로 `get_now() -> str` 함수를 만드세요 (현재 시각 반환).
# - `mcp_tools + [get_now]` 로 도구 리스트를 합쳐 새 에이전트를 만드세요.
# - "지금 몇 시인지 알려주고, 100 곱하기 7 도 계산해줘." 질문을 던져 두 도구가 모두 호출되는지 확인하세요.

# %%
from langchain.tools import tool
from datetime import datetime


@tool
def get_now() -> str:
    """현재 날짜와 시간을 반환합니다."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# MCP 도구 + 로컬 도구를 합쳐서 에이전트 생성
combined_tools = mcp_tools + [get_now]

hybrid_agent = create_agent(
    model,
    tools=combined_tools,
    system_prompt="당신은 도움이 되는 어시스턴트입니다. 필요한 도구를 골라 사용하세요.",
)

result = await hybrid_agent.ainvoke(
    {"messages": [{"role": "user", "content": "지금 몇 시인지 알려주고, 100 곱하기 7도 계산해줘."}]}
)
result["messages"][-1].pretty_print()

# %% [markdown]
# **관찰 포인트**
# - 한 질문 안에서 **출처가 다른 두 도구** (`get_now` = 로컬 `@tool`, `multiply` = MCP 서버)
#   가 함께 호출됩니다. 에이전트는 둘을 구분하지 않습니다.
# - 이것이 MCP 의 핵심 약속 — **표준 인터페이스 위에서 도구의 출처가 투명** 해진다는 것입니다.

# %% [markdown]
# ---
# ## 과제 9. 서버 프로세스 종료
#
# 실습이 끝나면 백그라운드로 띄운 MCP Server 를 종료합니다. 종료하지 않으면 포트 8000 이
# 계속 점유되어 다음 실행 시 충돌이 날 수 있습니다.
#
# **할 일**: `server_process.terminate()` 한 줄로 종료하세요.

# %%
server_process.terminate()
print("MCP Server를 종료했습니다.")

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 코드 | 학습 포인트 |
# |------|-----------|-------------|
# | ① Server 만들기 | `FastMCP` + `@mcp.tool()` | `@tool` 과 거의 동일한 사용법 |
# | ② Server 실행 | `subprocess.Popen([..., "math_server.py"])` | streamable_http 백그라운드 프로세스 |
# | ③ Client 연결 | `MultiServerMCPClient({"math": {...}})` | 여러 서버를 dict 로 등록 |
# | ④ 도구 변환 | `await client.get_tools()` | MCP 도구 → LangChain 도구 |
# | ⑤ 에이전트 통합 | `create_agent(model, tools=mcp_tools)` | 400 코드 재사용 — 변경 없음 |
# | ⑥ 도구 혼합 | `mcp_tools + [get_now]` | 외부 표준 + 내 전용 도구 함께 사용 |
#
# ### transport 방식
# - `stdio` : 데스크톱 앱이 로컬 서버를 하위 프로세스로 띄울 때 (노트북 환경에서는 제약 있음)
# - `streamable_http` : 서버를 HTTP 로 띄워 두고 URL 로 접속 — 노트북·원격 환경에 적합
#
# **핵심 메시지**:
# - MCP 는 도구·데이터 연결을 통일한 **오픈 표준** 이며 그 자체가 새로운 알고리즘은 아닙니다.
#   `@tool` 로 풀던 같은 문제를 "서버/클라이언트 분리" 라는 운영 패턴으로 푸는 것입니다.
# - 가장 큰 이득은 **재사용성과 호환성** — 한 번 만든 MCP Server 를 여러 모델·여러
#   프레임워크가 그대로 가져다 쓸 수 있습니다.
# - 에이전트 코드 입장에서는 도구의 출처가 무엇이든 인터페이스가 같으므로, MCP 도입은
#   **점진적·부분적** 으로 가능합니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `math_server.py` 에 `subtract(a, b)`, `divide(a, b)` 도구를 추가하고 서버를 재시작한 뒤,
#    "(100 - 25) 를 5 로 나누면?" 같은 질문에 새 도구들이 호출되는지 확인하세요.
# 2. 별도의 `weather_server.py` (400 의 `get_weather` 를 MCP 화) 를 만들어 포트 8001 에 띄우고,
#    `MultiServerMCPClient` 에 두 서버를 함께 등록해 "서울 기온 × 2 는?" 같은 질문에
#    **두 서버의 도구가 함께 호출** 되는지 관찰하세요.
# 3. `mcp_server.log` 파일을 일부러 비우고 서버 코드에 일부러 오류를 넣어, 실패 시 로그가
#    어떤 형태로 남는지 직접 보세요 — 디버깅 경로에 익숙해지는 것이 중요합니다.
# 4. (도전) `stdio` 방식의 MCP Server 도 만들어 보고, Claude Desktop 또는 VS Code 의 MCP
#    클라이언트 설정에 연결해 자기 도구를 IDE/데스크톱 앱에서 직접 호출해 보세요.
# 5. (도전) 공개 MCP 서버 (예: filesystem, github 등) 를 `MultiServerMCPClient` 에
#    추가로 등록해 에이전트가 외부 자원에 직접 접근하도록 확장해 보세요.

# %%
