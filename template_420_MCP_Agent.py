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
# # MCP Agent — 표준 프로토콜로 도구 연결하기
#
# **MCP (Model Context Protocol)** 는 Anthropic이 2024년 11월 공개한 오픈 표준으로,
# LLM(또는 에이전트)이 외부 도구·데이터에 연결되는 방식을 통일한 규약입니다.
# "AI를 위한 USB-C"라고 불립니다.
#
# - `400_Tools_Agents` 에서는 `@tool` 데코레이터로 **도구를 직접 정의**했습니다.
# - 이 노트북에서는 **MCP Server가 제공하는 도구를 그대로 가져와** 에이전트에 연결합니다.
#
# 즉, "도구를 만드는 사람"과 "도구를 쓰는 사람"이 분리됩니다.
# Server를 한 번 만들어 두면, MCP를 지원하는 모든 모델·프레임워크가 재사용할 수 있습니다.
#
# **참고**:
# - [MCP 공식 문서](https://modelcontextprotocol.io/)
# - [LangChain 공식 문서 - MCP](https://docs.langchain.com/oss/python/langchain/mcp)
#
# ## MCP의 구성
#
# | 역할 | 설명 |
# | ---- | ---- |
# | **Client** | 모델·에이전트 쪽. MCP Server에 접속해 도구를 가져옴 |
# | **Server** | 도구를 제공하는 쪽. Tools / Resources / Prompts 를 노출 |
#
# 이 노트북에서는 간단한 MCP Server를 직접 만들고, LangChain 에이전트를 Client로 연결합니다.

# %% [markdown]
# ## 0. 환경 설정
#
# Colab 등에서 실행 시 필요한 패키지를 설치합니다.
# - `langchain`, `langchain-google-genai` : 에이전트 및 Gemini 모델 (400 노트북과 동일)
# - `langchain-mcp-adapters` : MCP Server의 도구를 LangChain 도구로 변환
# - `mcp` : MCP Server를 만들기 위한 공식 Python SDK (HTTP 서버 실행 기능 포함)

# %%
# 필요한 라이브러리 설치 (Colab 등에서 최초 1회 실행)

# %%

# %%
# 모델 초기화 (400 노트북과 동일하게 Gemini 사용)

# %% [markdown]
# ## 1. MCP Server 만들기
#
# MCP Server는 도구를 제공하는 별도의 프로그램입니다.
# 공식 SDK의 `FastMCP`를 사용하면 `@mcp.tool()` 데코레이터만으로 손쉽게 만들 수 있습니다.
# (400 노트북의 `@tool` 데코레이터와 사용법이 거의 같습니다.)
#
# ### 연결 방식(transport) 두 가지
#
# | 방식 | 설명 |
# | ---- | ---- |
# | `stdio` | 데스크톱 앱(Claude Desktop, Cursor, VS Code 등)이 로컬 서버를 하위 프로세스로 띄울 때 사용 |
# | `streamable_http` | 서버를 HTTP로 띄워 두고 URL로 접속 (원격 서버에 접속하는 방식) |
#
# Colab/Jupyter 노트북에서는 `stdio` 방식이 표준 입출력 제약(`fileno` 미지원)으로
# 동작하지 않습니다. 따라서 이 노트북은 **`streamable_http`** 방식으로 서버를 띄우고 접속합니다.
#
# 아래 셀은 `math_server.py` 파일을 디스크에 직접 생성합니다.
# 이 파일 하나가 곧 독립적으로 실행되는 MCP Server가 됩니다.

# %%
# MCP Server 코드를 문자열로 정의한 뒤 파일로 저장합니다.
# MCP Server 객체 생성 ("Math"는 서버 이름, host/port 는 HTTP 접속 주소)
def add(a: int, b: int) -> int:
def multiply(a: int, b: int) -> int:
    # streamable-http 방식으로 서버 실행 (기본 주소: http://127.0.0.1:8000/mcp)
    # 주의: MCP SDK 의 transport 이름은 하이픈("streamable-http")입니다.
    #       클라이언트(langchain-mcp-adapters)는 언더스코어("streamable_http")를 쓰므로 혼동 주의.


# %% [markdown]
# ## 2. MCP Server 실행
#
# `streamable_http` 서버는 실행되면 계속 떠 있어야 하므로, 노트북 셀을 막지 않도록
# **별도의 백그라운드 프로세스**로 띄웁니다.
# 서버의 출력(로그·오류)은 `mcp_server.log` 파일로 보냅니다 — 문제가 생기면 이 파일을 확인하세요.

# %%
# 이미 실행 중인 서버가 있으면 먼저 종료합니다 (셀 재실행 대비).
# MCP Server를 백그라운드 프로세스로 실행 (출력은 로그 파일로)
def wait_for_server(host="127.0.0.1", port=8000, timeout=30):
# 서버가 포트를 열 때까지 대기

# %% [markdown]
# ## 3. MCP Client로 서버에 연결
#
# `MultiServerMCPClient`는 여러 MCP Server에 동시에 연결할 수 있는 Client입니다.
# 각 서버를 딕셔너리로 등록하며, 여기서는 `streamable_http` 방식으로
# 방금 띄운 서버의 주소를 지정합니다.

# %%
# 연결할 MCP Server 목록 정의
        # 다른 서버를 추가하려면 여기에 항목을 더 넣으면 됩니다.


# %% [markdown]
# ## 4. MCP 도구 가져오기
#
# `get_tools()`는 등록된 모든 MCP Server에 접속해, 서버가 제공하는 도구들을
# **LangChain 도구 객체로 변환**해서 돌려줍니다.
#
# `get_tools()`는 비동기(async) 함수이므로 `await`로 호출합니다.
# (Jupyter 노트북 셀은 top-level `await`를 지원합니다.)

# %%
# MCP Server의 도구를 LangChain 도구로 변환

# %% [markdown]
# ## 5. MCP 도구로 에이전트 생성
#
# 변환된 도구는 LangChain의 일반 도구와 동일하므로,
# 400 노트북에서 쓴 `create_agent`를 **그대로** 사용할 수 있습니다.
# MCP 도구든 `@tool` 도구든 에이전트 입장에서는 차이가 없습니다.

# %%

# %% [markdown]
# ## 6. 에이전트 실행
#
# MCP 도구는 내부적으로 비동기로 동작하므로, 노트북에서는 비동기 API
# (`ainvoke`, `astream`)로 호출하는 것이 안전합니다.

# %%
# 에이전트 호출 (비동기)

# %% [markdown]
# ### 추론 과정 확인 (스트리밍)
#
# `astream`으로 실행하면 에이전트가 어떤 MCP 도구를 어떤 인수로 호출하는지
# 단계별로 확인할 수 있습니다.

# %%

# %% [markdown]
# ## 7. MCP 도구와 로컬 도구 함께 쓰기
#
# MCP 도구와 `@tool`로 직접 만든 로컬 도구를 **섞어서** 하나의 에이전트에 넘길 수 있습니다.
# 외부 표준 도구(MCP)와 내가 만든 전용 도구를 함께 활용하는 실전 패턴입니다.

# %%
def get_now() -> str:
# MCP 도구 + 로컬 도구를 합쳐서 에이전트 생성

# %% [markdown]
# ## 8. 서버 프로세스 종료
#
# 실습이 끝나면 백그라운드로 띄운 MCP Server를 종료합니다.

# %%

# %% [markdown]
# ## 주요 포인트 정리
#
# ### MCP Agent
# 1. **MCP란**: 도구·데이터 연결을 통일한 오픈 표준 ("AI를 위한 USB-C")
# 2. **Server 만들기**: `FastMCP` + `@mcp.tool()` — `@tool`과 거의 동일한 사용법
# 3. **Server 실행**: `streamable_http` 방식으로 백그라운드 프로세스로 기동
# 4. **Client 연결**: `MultiServerMCPClient`로 서버 URL 등록
# 5. **도구 변환**: `await client.get_tools()` 로 MCP 도구를 LangChain 도구로 변환
# 6. **에이전트 통합**: `create_agent`는 그대로 재사용 — MCP 도구든 로컬 도구든 동일하게 취급
#
# ### transport 방식
# - `stdio` : 데스크톱 앱이 로컬 서버를 하위 프로세스로 띄울 때 (노트북 환경에서는 제약 있음)
# - `streamable_http` : 서버를 HTTP로 띄워 두고 URL로 접속 — 노트북·원격 환경에 적합
#
# ### `@tool` (400) vs MCP (420)
# | 구분 | `@tool` 도구 | MCP 도구 |
# | ---- | ----------- | -------- |
# | 정의 위치 | 에이전트 코드 안 | 별도 Server 프로세스 |
# | 재사용 | 해당 코드에서만 | 표준 프로토콜로 어디서나 |
# | 적합한 경우 | 프로젝트 전용 간단한 도구 | 공용·외부 제공 도구 |

# %%
