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
# # RAG (Retrieval Augmented Generation) 실습 - Gemini
#
# RAG는 LLM이 내부 지식만으로 답하는 방식이 아니라, 질문과 관련된 문서를 추가로 제공받아 그것을 근거로 답변을 생성하는 방법입니다.
#
# ## RAG Flow
# 1. **Document Ingestion**: 문서를 작은 청크(chunk)로 분할하고 임베딩으로 변환하여 벡터 저장소에 저장
# 2. **Query Processing**: 사용자 질문을 임베딩으로 변환
# 3. **Retrieval**: 질문 임베딩과 유사한 문서 청크를 검색
# 4. **Generation**: 검색된 문서와 질문을 함께 LLM에 제공하여 답변 생성
#
# ## 사용 기술
# - **임베딩 모델**: gemini-embedding-001 (Gemini Embedding API)
# - **생성 모델**: Gemini (gemini-2.5-flash)

# %%
# 환경 변수 로드
# Gemini API 클라이언트 초기화

# %% [markdown]
# ## 1. Source: 문서 데이터 준비
#
# RAG 시스템의 첫 단계는 다양한 소스에서 문서를 준비하는 것입니다.
# 실제 환경에서는 PDF, 웹사이트, 데이터베이스 등에서 문서를 로드합니다.
# Text, PPT, Image, PDF, HTML 등 다양한 비정형 데이터 소스를 지원합니다.

# %%
# 샘플 문서 데이터 (실제로는 외부 소스에서 로드)

# %% [markdown]
# ## 2. Load & Transform: 문서를 청크로 분할
#
# 다양한 소스(Web Site, DB, YouTube 등)에서 문서(HTML, PDF, JSON, Word, PPT, 코드 등)를 로드합니다.
# 긴 문서는 작은 단위(청크)로 나누어야 합니다. 이렇게 하면:
# - 임베딩 생성이 효율적입니다
# - 검색 시 더 정확한 관련 부분을 찾을 수 있습니다
# - LLM의 컨텍스트 길이 제한을 고려할 수 있습니다
#
# 데이터 변환 및 정제를 위한 여러 변환 단계를 포함합니다.

# %%
def split_into_chunks(text, chunk_size=200, overlap=50):
# 모든 문서를 청크로 분할


# %%

# %% [markdown]
# ## 3. Embed: 문서를 임베딩으로 변환
#
# 각 문서 청크를 숫자 벡터(임베딩)로 변환합니다.
# 유사한 의미를 가진 텍스트는 유사한 벡터로 표현됩니다.
# 문서에 대한 임베딩을 만들어 유사한 다른 텍스트 부분을 빠르고 효율적으로 검색할 수 있습니다.
# **gemini-embedding-001 모델**을 사용하여 임베딩을 생성합니다.

# %%
def embed_texts(texts, model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"):
    # 각 임베딩을 numpy 배열로 변환
# 모든 청크를 임베딩으로 변환


# %% [markdown]
# ## 4. Store: 임베딩을 벡터 저장소에 저장
#
# 생성된 임베딩을 벡터 저장소에 저장합니다.
# 실제 환경에서는 Pinecone, Weaviate, ChromaDB 등의 벡터 데이터베이스를 사용합니다.
# 임베딩의 효율적인 저장 및 검색을 지원하는 데이터베이스입니다.

# %%
# 벡터 저장소 (실제로는 벡터 DB를 사용)

# %% [markdown]
# ## 5. Retrieve: 질문과 유사한 문서 검색
#
# 사용자 질문을 임베딩으로 변환하고, 저장된 문서 임베딩과의 유사도를 계산하여
# 가장 관련성 높은 문서를 검색합니다.
# 검색 알고리즘을 통해 문서 유사도를 측정합니다.

# %%
def search_relevant_chunks(query, vector_store, top_k=3):
    # 1. 질문을 임베딩으로 변환
    # 2. 모든 청크와의 유사도 계산
    # 3. 상위 k개 선택
    # 4. 결과 반환
# 검색 테스트


# %% [markdown]
# ## 6. RAG 시스템 구현
#
# 검색된 문서를 컨텍스트로 사용하여 Gemini API로 답변을 생성합니다.
# 이것이 RAG의 핵심입니다: 검색(Retrieval) + 생성(Generation)

# %%
def rag_query(query, vector_store, top_k=3):
    # 1. 관련 문서 검색
    # 2. 컨텍스트 구성
    # 3. 프롬프트 구성
    # 4. Gemini API로 답변 생성

# %% [markdown]
# ## 7. RAG 시스템 테스트
#
# 다양한 질문으로 RAG 시스템을 테스트해봅시다.

# %%
# 테스트 쿼리들
    # RAG로 답변 생성
    # 검색된 문서 출력
    # 생성된 답변 출력

# %% [markdown]
# ## 8. RAG vs 일반 LLM 비교
#
# RAG를 사용하면 LLM이 최신 정보나 특정 도메인 지식을 활용할 수 있습니다.
# 일반 LLM과 비교해봅시다.

# %%

# %% [markdown]
# ## 요약
#
# RAG 시스템은 다음 단계로 구성됩니다:
#
# 1. **Source**: 다양한 소스에서 문서 수집 (Text, PPT, Image, PDF, HTML 등)
# 2. **Load**: 다양한 소스에서 문서 로드 (Web Site, DB, YouTube 등)
# 3. **Transform**: 문서를 적절한 크기의 청크로 분할 (데이터 변환 및 정제)
# 4. **Embed**: 각 청크를 임베딩 벡터로 변환 (gemini-embedding-001 사용, 유사한 텍스트 부분을 빠르고 효율적으로 검색)
# 5. **Store**: 임베딩을 벡터 저장소에 저장 (효율적인 저장 및 검색을 지원하는 데이터베이스)
# 6. **Retrieve**: 질문과 유사한 문서를 검색 (검색 알고리즘을 통한 문서 유사도 측정)
# 7. **Prompt**: 검색된 문서와 질문을 결합하여 프롬프트 생성
# 8. **LLM**: Gemini 모델로 답변 생성
# 9. **Answer**: 최종 답변 반환
#
# 이 과정을 통해 LLM은 최신 정보와 특정 도메인 지식을 활용하여 더 정확한 답변을 생성할 수 있습니다.
#
# ## 사용 기술
# - **임베딩**: gemini-embedding-001 (Gemini Embedding API)
# - **생성 모델**: Gemini (gemini-2.5-flash)

# %%
