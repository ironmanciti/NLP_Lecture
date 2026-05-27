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
# # 400. 수강생 실습 — LangChain Tools & Agents 기초
#
# ## 학습 목표
# LLM 이 자기 머릿속 지식만으로 답하는 것을 넘어, **외부 함수(도구) 를 직접 호출해서
# 그 결과로 답변하도록** 만드는 LangChain 의 Tools & Agents 개념을 처음부터 익힙니다.
#
# ## 왜 Tools & Agents 인가?
# - LLM 은 **계산기**가 아닙니다 — "1234 × 5678" 같은 계산은 환각이 잦습니다.
# - LLM 은 **실시간 정보** 를 모릅니다 — "지금 서울 기온" 은 사전학습 시점 이후 데이터입니다.
# - LLM 은 **사내 DB** 를 모릅니다 — 고객 정보·재고·예약 같은 데이터는 외부 함수로만 알 수 있습니다.
#
# 도구(Tools) 는 "LLM 이 호출할 수 있는 일반 함수" 이고, 에이전트(Agent) 는
# **"질문을 보고 어떤 도구를 어떤 인자로 호출할지 스스로 결정"** 하는 시스템입니다.
#
# ## ReAct 패턴 — Agent 가 생각하는 방식
# ```
#   질문 → [Reason: 어떤 도구 필요?] → [Act: 도구 호출] → [Observe: 결과 확인]
#                                                              │
#                                  └────── 더 필요하면 반복 ────┘
#                                                              │
#                                          최종 답변 생성 ─────┘
# ```
#
# ## 사용 기술
# - **LLM**: Google Gemini (`gemini-2.5-flash`)
# - **프레임워크**: LangChain v1 (`create_agent`, `@tool`)
# - **외부 API 예시**: Open-Meteo (날씨)
#
# ## 참고 링크
# - [LangChain 공식 문서 - Tools](https://docs.langchain.com/oss/python/langchain/tools)
# - [LangChain 공식 문서 - Agents](https://docs.langchain.com/oss/python/langchain/agents)

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.
#
# - `langchain` : `create_agent` 등 v1 API (v1.0 이상 필요)
# - `langchain-google-genai` : `model_provider="google_genai"` 사용에 필요
# - `python-dotenv` : `.env` 에서 API 키 로드

# %%
# !pip install -q -U langchain langchain-google-genai python-dotenv

# %% [markdown]
# ---
# ## 과제 1. 환경 변수 로드 + Gemini 모델 초기화
#
# `.env` 의 `GOOGLE_API_KEY` 를 자동으로 환경 변수에 올린 뒤, LangChain 의
# `init_chat_model` 로 Gemini 모델을 만듭니다.
#
# **할 일**:
# - `python-dotenv` 의 `load_dotenv()` 로 `.env` 를 로드하세요.
# - `init_chat_model("gemini-2.5-flash", model_provider="google_genai")` 로 모델을 만드세요.
#
# **힌트**: Gemini API 키는 https://aistudio.google.com/apikey 에서 발급받아
# 프로젝트 루트 `.env` 에 `GOOGLE_API_KEY=...` 형태로 저장해 두면 됩니다.

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %%
from langchain.chat_models import init_chat_model

# 모델 초기화
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
model = init_chat_model("gemini-3.5-flash", model_provider="google_genai")
model

# %% [markdown]
# ---
# ## 과제 2. 첫 번째 도구 정의 (`@tool` 데코레이터)
#
# 도구(Tool) 는 모델이 호출할 수 있는 일반 함수입니다. **type hint 와 docstring** 이
# 도구의 **입력 스키마와 설명** 이 되므로 둘 다 반드시 정확히 작성해야 합니다 — LLM 은
# 이 정보를 보고 도구를 언제 어떤 인자로 호출할지 결정합니다.
#
# **할 일**:
# - `@tool` 데코레이터를 붙여 `search_db(query: str, limit: int = 10) -> str` 함수를 만드세요.
# - docstring 에 도구의 목적과 각 인자의 의미를 작성하세요.
# - `.name` 과 `.description` 속성으로 LangChain 이 어떻게 메타데이터를 추출했는지 확인하세요.
#
# **힌트**: type hint 가 없으면 도구가 동작하지 않습니다 — LLM 이 인자의 자료형을 모르기 때문입니다.

# %%
from langchain.tools import tool


# 가장 간단하게 도구를 만드는 방법은 @tool 데코레이터를 사용하는 것입니다.
# Type hints 는 필수입니다. 이들은 도구의 입력 스키마(input schema)를 정의하기 때문입니다.
# 독스트링(docstring)은 모델이 도구의 목적을 이해할 수 있도록 간결하면서도 유용한 정보를 포함해야 합니다.

@tool
def search_db(query: str, limit: int = 10) -> str:
    """검색어(query)에 해당하는 고객 데이터베이스 레코드를 조회합니다.

    Args:
        query: 검색할 키워드 또는 문장
        limit: 반환할 최대 결과 개수
    """
    return f"'{query}'에 대한 검색 결과 {limit}개를 찾았습니다."


# 도구 정보 확인
print("도구 이름:", search_db.name)
print("도구 설명:", search_db.description)

# %% [markdown]
# **관찰 포인트**
# - 도구 이름은 함수 이름(`search_db`) 에서 자동 생성됩니다.
# - 도구 설명은 docstring 의 첫 줄에서 추출됩니다 — 그래서 docstring 의 **첫 줄이 가장 중요** 합니다.
# - LLM 이 보는 정보는 결국 이 (이름 + 설명 + 인자 type hint) 세 가지뿐입니다.

# %% [markdown]
# ---
# ## 과제 3. 도구 이름/설명 사용자 정의
#
# 함수 이름이 LLM 입장에서 명확하지 않을 때는 도구 이름/설명을 명시적으로 지정합니다.
#
# **할 일**:
# - `@tool("calculator", description="...")` 형식으로 데코레이터에 이름과 설명을 지정하세요.
# - 함수 본체에서 `eval(expression)` 으로 수식 평가를 수행하세요 (실패하면 오류 메시지 반환).
#
# **힌트**:
# - LLM 은 "이 도구를 언제 써야 하나?" 를 description 만 보고 판단합니다.
# - "수학 문제를 풀 때 이 도구를 사용하세요" 처럼 **사용 시나리오** 를 명시하면
#   모델의 도구 선택 정확도가 올라갑니다.

# %%
@tool("calculator", description="산술 계산을 수행합니다. 수학 문제를 풀 때 이 도구를 사용하세요.")
def calc(expression: str) -> str:
    """수학 표현식을 계산합니다."""
    print("\n--- calculator 도구 실행됨 ---")
    try:
        return str(eval(expression))
    except Exception as e:
        return f"계산 오류: {e}"


print("도구 이름:", calc.name)
print("도구 설명:", calc.description)

# %% [markdown]
# **관찰 포인트**
# - 함수 이름은 `calc` 이지만 LangChain 에 등록된 도구 이름은 `calculator` 입니다.
# - 함수의 docstring 보다 데코레이터의 `description` 이 우선합니다.
# - `print("... calculator 도구 실행됨 ...")` 은 나중에 Agent 가 이 도구를 실제로
#   호출했는지를 눈으로 확인하기 위한 디버그 흔적입니다.

# %% [markdown]
# ---
# ## 과제 4. Pydantic 스키마로 고급 도구 정의 (Open-Meteo 날씨 API)
#
# 인자가 여러 개거나 의미를 명확히 전달해야 할 때는 **Pydantic** 모델로 입력 스키마를
# 따로 정의합니다. 각 필드의 `Field(description=...)` 가 LLM 에게 인자의 의미를 알려줍니다.
#
# **할 일**:
# - `WeatherInput` Pydantic 모델로 `latitude`, `longitude` 두 필드를 정의하세요.
# - `@tool(args_schema=WeatherInput)` 데코레이터로 도구를 만드세요.
# - `requests.get(...)` 로 Open-Meteo API 를 호출해 현재 기온을 반환하세요.
# - 서울 좌표(37.56667, 126.97806) 로 테스트하세요.
#
# **힌트**: Open-Meteo 는 API 키 없이 사용할 수 있는 무료 날씨 API 입니다.
# 응답 JSON 의 `current.temperature_2m` 필드에 섭씨 온도가 들어 있습니다.

# %%
from pydantic import BaseModel, Field
from langchain_core.tools import tool
import requests


# 입력 데이터 구조 정의 (Pydantic 사용)
class WeatherInput(BaseModel):
    """날씨 질의에 사용할 입력 스키마"""
    latitude: float = Field(description="질의할 지역의 위도를 입력합니다.")
    longitude: float = Field(description="질의할 지역의 경도를 입력합니다.")


# 현재의 온도 가져오기
@tool(args_schema=WeatherInput)
def get_weather(latitude: float, longitude: float) -> str:
    """
    제공된 좌표의 현재 기온을 섭씨(Celsius) 단위로 가져옵니다.
    """
    print('get_weather 도구 호출됨')
    try:
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m")
        data = response.json()
        temp = data['current']['temperature_2m']
        return f"현재 기온: {temp}°C"
    except Exception as e:
        return f"날씨 정보를 가져오는 중 오류 발생: {e}"


# 서울의 위도, 경도로 테스트
print(get_weather.invoke({'latitude': 37.56667, 'longitude': 126.97806}))

# %% [markdown]
# **관찰 포인트**
# - `get_weather.invoke({...})` 처럼 직접 호출도 가능합니다 — 도구가 일반 함수일 뿐임을 보여줍니다.
# - LLM 은 "서울 위도/경도" 같은 상식을 **자기 지식으로** 알고 있으므로, 사용자가
#   "서울 기온?" 만 물어봐도 적절한 위경도를 채워 넣어 도구를 호출할 수 있습니다 — 다음 과제에서 확인.

# %% [markdown]
# ---
# ## 과제 5. ReAct Agent 생성 (`create_agent`)
#
# 도구 3개를 LLM 에 묶어 "스스로 어떤 도구를 쓸지 결정하는 에이전트" 를 만듭니다.
# LangChain v1 의 `create_agent(model, tools)` 한 줄로 ReAct 에이전트가 완성됩니다.
#
# **할 일**:
# - `available_tools = [search_db, calc, get_weather]` 로 도구 리스트를 만드세요.
# - `create_agent(model=model, tools=available_tools)` 로 에이전트를 만드세요.
# - 에이전트가 사용할 수 있는 도구 이름 목록을 출력하세요.
#
# **힌트**: `create_agent` 는 내부적으로 LangGraph 의 상태 머신을 만들어
# (Reason → Act → Observe → ...) 반복 루프를 자동으로 돌려줍니다.

# %%
from langchain.agents import create_agent

# 사용 가능한 도구 목록 준비
available_tools = [search_db, calc, get_weather]

# ReAct 에이전트 생성
agent = create_agent(
    model=model,
    tools=available_tools  # Agent 가 사용할 도구 목록
)

print("에이전트가 생성되었습니다.")
print(f"에이전트가 사용할 수 있는 도구: {[tool.name for tool in available_tools]}")
agent

# %% [markdown]
# **관찰 포인트**
# - 도구 목록을 늘리거나 줄이는 것만으로 에이전트의 능력을 확장/축소할 수 있습니다.
# - 동일한 모델·동일한 시스템 프롬프트라도 **도구 설명을 잘 쓴 도구** 가 더 자주 선택됩니다.

# %% [markdown]
# ---
# ## 과제 6. Agent 호출 (invoke)
#
# 에이전트는 메시지 시퀀스를 입력으로 받습니다. system + user 메시지를 넘기면
# 에이전트가 알아서 도구를 골라 호출하고 최종 답변까지 만들어 옵니다.
#
# **할 일**:
# - "지금 서울 기온이 몇도인가요?" 질문으로 `agent.invoke({...})` 를 호출하세요.
# - `result['messages'][-1].pretty_print()` 로 마지막 메시지(최종 답변) 를 출력하세요.
#
# **힌트**:
# - 결과의 `messages` 에는 [system, user, AI tool_call, tool result, AI final answer]
#   같은 시퀀스가 들어 있습니다 — 한 번 전체를 출력해 ReAct 흐름을 직접 확인해 보세요.
# - 모델이 자체적으로 "서울 = 위도 37.57, 경도 126.98" 으로 결정해 `get_weather` 를 호출해야 정상입니다.

# %%
# 기본 에이전트 호출 예제
result = agent.invoke(
    {"messages": [
        {'role': 'system', "content": "당신은 도움이 되는 어시스턴트입니다. 주어진 도구를 이용해 답변하세요."},
        {"role": "user", "content": "지금 서울 기온이 몇도인가요?"}
    ]}
)

print("에이전트 응답:")
result['messages'][-1].pretty_print()

# %% [markdown]
# **관찰 포인트**
# - 도구 4 의 `print('get_weather 도구 호출됨')` 이 출력되었다면 → 에이전트가 실제로
#   날씨 도구를 골라 호출했다는 증거입니다.
# - 모델은 위경도를 사용자에게 되묻지 않고 **자기 상식**(서울 = 약 37.57, 126.98) 으로
#   채워 넣었습니다. 이것이 도구 호출의 가장 강력한 점 — LLM 의 일반 상식과 외부
#   함수의 정확한 데이터가 자연스럽게 결합됩니다.

# %% [markdown]
# ---
# ## 과제 7. 시스템 프롬프트로 행동 제어
#
# 에이전트의 톤·답변 길이·작업 방식을 제어하고 싶다면 `system_prompt` 를 직접 넘깁니다.
# 매 invoke 마다 시스템 메시지를 보내는 대신 에이전트 자체에 한 번만 박아 둘 수 있습니다.
#
# **할 일**:
# - `create_agent(..., system_prompt="...")` 형태로 새 에이전트를 만드세요.
# - "5 더하기 3은?" 질문으로 호출해, 모델이 `calculator` 도구를 호출하는지 확인하세요.
#
# **힌트**: 단순 계산이라 모델이 도구 없이 직접 답할 수도 있고, calculator 도구를
# 호출할 수도 있습니다. 도구를 호출하면 과제 3 의 `--- calculator 도구 실행됨 ---`
# 메시지가 출력됩니다.

# %%
agent_with_prompt = create_agent(
    model=model,
    tools=[search_db, calc, get_weather],
    system_prompt="당신은 도움이 되는 어시스턴트입니다. 간결하고 정확하게 답변하세요."
)

result = agent_with_prompt.invoke(
    {"messages": [{"role": "user", "content": "5 더하기 3은?"}]}
)

print("시스템 프롬프트가 적용된 에이전트 응답:")
result['messages'][-1].pretty_print()

# %% [markdown]
# **관찰 포인트**
# - "5 더하기 3" 처럼 간단한 계산은 모델이 도구 없이 바로 답할 수도 있습니다.
# - 더 어려운 식("(173 * 41) + 7892 / 4 는?") 을 던지면 calculator 도구가 호출될 확률이 올라갑니다.
# - 도구 호출 여부는 모델·프롬프트·도구 설명에 따라 달라지므로 **결정론적이지 않습니다**.

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 코드 | 학습 포인트 |
# |------|-----------|-------------|
# | ① 도구 정의 | `@tool` 데코레이터 + type hint + docstring | LLM 은 이 3가지 메타정보로 도구를 선택 |
# | ② 이름/설명 재정의 | `@tool("name", description=...)` | "언제 써야 하는가" 를 description 에 명시 |
# | ③ 복잡한 스키마 | Pydantic `BaseModel` + `Field(description=...)` | 다중 인자나 복잡한 입력에 사용 |
# | ④ 에이전트 생성 | `create_agent(model, tools=[...])` | ReAct 루프가 자동으로 동작 |
# | ⑤ 에이전트 호출 | `agent.invoke({"messages": [...]})` | 메시지 시퀀스 입출력 패턴 |
# | ⑥ 행동 제어 | `system_prompt="..."` | 톤·답변 길이·작업 방식을 한 번에 지정 |
#
# **핵심 메시지**:
# - 도구는 결국 **잘 문서화된 평범한 함수** 입니다 — 특별한 것은 type hint 와 docstring
#   을 LLM 이 메타정보로 활용한다는 점뿐.
# - 에이전트의 능력은 **연결된 도구의 품질** 이 결정합니다. 도구가 부정확하거나
#   설명이 부실하면 에이전트도 부정확해집니다.
# - LLM 의 "상식 + 추론" 과 도구의 "정확한 데이터" 가 결합되는 지점이 Agent 의 강점입니다.
#
# **실전 예제**: 이 노트북을 학습한 후 `streamlit-llm_LangChain/060_Agent.py` 를 실행해
# Streamlit 챗봇에 에이전트를 붙이는 예제도 살펴보세요.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. 새 도구 `get_current_time() -> str` 을 만들어 (`datetime.now().isoformat()` 반환)
#    에이전트에 추가하고, "지금 몇시야?" 질문을 던져 보세요.
# 2. `search_db` 의 description 을 "고객 정보를 찾을 때만 사용" 으로 더 구체적으로
#    바꾼 뒤, 일반 검색 질문("RAG 란?") 에서는 호출하지 않는지 확인하세요.
# 3. `calc` 의 `eval` 호출을 **`ast.literal_eval`** 또는 화이트리스트 기반의 안전한
#    수식 평가기로 교체해 보세요 — `eval` 은 사용자 입력에 직접 노출되면 위험합니다.
# 4. 에이전트 호출 결과의 `result['messages']` 전체를 순회하며 (tool_call 메시지,
#    tool result 메시지, AI 최종 답변) 흐름을 출력해 ReAct 루프 한 사이클을 시각적으로 확인하세요.
# 5. (도전) `get_weather` 에 도시명을 받아서 위경도로 변환하는 두 번째 도구를 추가하고,
#    "도쿄 기온?" 같은 질문에서 두 도구가 **순차적으로** 호출되는지 관찰하세요.

# %%
