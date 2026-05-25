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
# # RAG Agent (검색 증강 생성 에이전트)
#
# **RAG (Retrieval-Augmented Generation)** 는 외부 지식 소스에서 정보를 검색하여 LLM의 응답을 향상시키는 기술입니다.
#
# RAG 에이전트는 검색 도구를 사용하여 관련 문서를 찾고, 이를 컨텍스트로 활용하여 답변을 생성합니다.
#
# **참고**: [LangChain 공식 문서 - RAG Agent](https://docs.langchain.com/oss/python/langchain/rag)
#
# ## RAG 워크플로우
#
# 1. **인덱싱 (Indexing)**: 문서를 로드, 분할, 임베딩하여 벡터 스토어에 저장
# 2. **검색 및 생성 (Retrieval & Generation)**: 사용자 질문에 대해 관련 문서를 검색하고 답변 생성
#
# 에이전트가 필요할 때만 검색 도구를 호출합니다.
# - 장점: 유연성, 필요할 때만 검색
# - 단점: 여러 번의 모델 호출로 인한 지연 시간

# %%
# 필요한 라이브러리 설치 (Colab 등에서 최초 1회 실행)

# %%

# %%
# 모델 및 임베딩 초기화 (Gemini 사용)
# 벡터 스토어 생성

# %% [markdown]
# ## 1. 인덱싱 단계
#
# 문서를 로드하고 벡터 스토어에 저장합니다.

# %% [markdown]
# ### 1.1 문서 로드 및 분할

# %%
# 웹 문서 로드
# 텍스트 분할

# %% [markdown]
# ### 1.2 벡터 스토어에 저장

# %%
# 문서를 벡터 스토어에 추가

# %% [markdown]
# ### 2. 체인 기반 RAG (Chain-based RAG)
#
# 검색을 먼저 실행하고 검색 결과를 LLM에 컨텍스트로 제공

# %%
# 체인 기반 RAG 예제
def chain_based_rag(query: str):
    # 1. 검색
    # 2. 컨텍스트와 함께 답변 생성
# 체인 기반 RAG 테스트


# %% [markdown]
# ### 3. 에이전트 기반 RAG (Agentic RAG)

# %% [markdown]
# ### 3.1 검색 도구 생성

# %%
def retrieve_context(query: str):
    # 벡터 스토어에서 유사한 문서 검색
    # 검색된 문서를 문자열로 직렬화
    # 문자열과 문서 객체를 함께 반환
# 도구 테스트

# %% [markdown]
# ### 3.2 RAG 에이전트 생성

# %%
# 시스템 프롬프트 설정
# 에이전트 생성

# %% [markdown]
# ### 3.3 에이전트 실행

# %%
# 질문에 대한 답변 생성
# query = "에이전트의 주요 특징은 무엇인가요?"
# query = "프롬프트 엔지니어링이 중요한 이유는 무엇인가요?"
# 스트리밍 방식으로 실행

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **인덱싱**: 문서 로드 → 분할 → 임베딩 → 벡터 스토어 저장
# 2. **검색 도구**: 벡터 스토어를 래핑한 도구 생성
# 3. **에이전트 통합**: 검색 도구를 에이전트에 추가
# 4. **에이전트 기반 vs 체인 기반**: 필요에 따라 선택

# %%
