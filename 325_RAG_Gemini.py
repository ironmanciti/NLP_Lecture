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
from google import genai
from google.genai import types
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 환경 변수 로드
load_dotenv()

# Gemini API 클라이언트 초기화
client = genai.Client()

# %% [markdown]
# ## 1. Source: 문서 데이터 준비
#
# RAG 시스템의 첫 단계는 다양한 소스에서 문서를 준비하는 것입니다.
# 실제 환경에서는 PDF, 웹사이트, 데이터베이스 등에서 문서를 로드합니다.
# Text, PPT, Image, PDF, HTML 등 다양한 비정형 데이터 소스를 지원합니다.

# %%
# 샘플 문서 데이터 (실제로는 외부 소스에서 로드)
documents = [
    "인공지능(AI)은 인간의 진화 과정에서 자연 발생한 생물학적 지식입니다. 머신러닝과 딥러닝은 AI의 하위 분야로, 대량의 데이터를 통해 패턴을 학습합니다.",
    "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 AI의 한 분야입니다. 텍스트 분석, 번역, 감성 분석, 챗봇 등 다양한 응용 분야가 있습니다.",
    "Transformer는 Google에서 만들어낸 인공지능 모델로, 어텐션 메커니즘을 핵심으로 합니다. BERT, GPT 등 최신 언어 모델의 기반이 되었습니다.",
    "RAG(Retrieval Augmented Generation)는 외부 지식 베이스에서 질문과 관련된 정보를 검색하여 LLM의 답변을 보강하는 기법입니다. 먼저 문서를 작은 청크로 나누고 각 청크를 임베딩 벡터로 변환해 벡터 데이터베이스에 저장합니다. 질문이 들어오면 질문을 임베딩으로 바꾼 뒤 유사도가 높은 청크를 검색하고, 검색된 청크를 컨텍스트로 함께 제공하여 답변을 생성합니다. 이를 통해 모델이 학습하지 못한 최신 정보나 특정 도메인 지식에도 접근할 수 있으며, 환각 현상을 줄여 답변의 정확도와 신뢰성을 크게 높일 수 있습니다.",
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다. 임베딩 벡터를 저장하고 유사도 검색에 활용됩니다."
]

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
    """
    긴 텍스트를 청크로 분할

    Args:
        text: 분할할 텍스트
        chunk_size: 각 청크의 크기 (문자 수)
        overlap: 청크 간 겹치는 부분 (문자 수)

    Returns:
        list: 청크 리스트
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks

# 모든 문서를 청크로 분할
all_chunks = []
for i, doc in enumerate(documents):
    chunks = split_into_chunks(doc, chunk_size=150, overlap=30)
    all_chunks.extend(chunks)
    print(f"문서 {i+1}이 {len(chunks)}개의 청크로 분할되었습니다.")

print(f"\n총 {len(all_chunks)}개의 청크가 생성되었습니다.")

# %%
all_chunks


# %% [markdown]
# ## 3. Embed: 문서를 임베딩으로 변환
#
# 각 문서 청크를 숫자 벡터(임베딩)로 변환합니다.
# 유사한 의미를 가진 텍스트는 유사한 벡터로 표현됩니다.
# 문서에 대한 임베딩을 만들어 유사한 다른 텍스트 부분을 빠르고 효율적으로 검색할 수 있습니다.
# **gemini-embedding-001 모델**을 사용하여 임베딩을 생성합니다.

# %%
def embed_texts(texts, model="gemini-embedding-001", task_type="RETRIEVAL_DOCUMENT"):
    """
    텍스트 리스트를 임베딩 벡터로 변환

    Args:
        texts: 임베딩할 텍스트 리스트
        model: 사용할 임베딩 모델
        task_type: 작업 유형 ("SEMANTIC_SIMILARITY", "RETRIEVAL_QUERY", "RETRIEVAL_DOCUMENT" 등)

    Returns:
        numpy array: 임베딩 벡터 행렬
    """
    result = client.models.embed_content(
        model=model,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type)
    )

    # 각 임베딩을 numpy 배열로 변환
    embeddings = np.array([np.array(e.values) for e in result.embeddings])

    return embeddings

# 모든 청크를 임베딩으로 변환
print("청크를 임베딩으로 변환 중...")
chunk_embeddings = embed_texts(all_chunks, task_type="RETRIEVAL_DOCUMENT")

print(f"\n임베딩 완료! Shape: {chunk_embeddings.shape}")
print(f"각 청크는 {chunk_embeddings.shape[1]}차원 벡터로 표현됩니다.")

# %% [markdown]
# ## 4. Store: 임베딩을 벡터 저장소에 저장
#
# 생성된 임베딩을 벡터 저장소에 저장합니다.
# 실제 환경에서는 Pinecone, Weaviate, ChromaDB 등의 벡터 데이터베이스를 사용합니다.
# 임베딩의 효율적인 저장 및 검색을 지원하는 데이터베이스입니다.

# %%
# 벡터 저장소 (실제로는 벡터 DB를 사용)
vector_store = {
    'chunks': all_chunks,
    'embeddings': chunk_embeddings
}

print("벡터 저장소에 저장 완료!")
print(f"- 저장된 청크 수: {len(vector_store['chunks'])}")
print(f"- 임베딩 벡터 shape: {vector_store['embeddings'].shape}")

# %% [markdown]
# ## 5. Retrieve: 질문과 유사한 문서 검색
#
# 사용자 질문을 임베딩으로 변환하고, 저장된 문서 임베딩과의 유사도를 계산하여
# 가장 관련성 높은 문서를 검색합니다.
# 검색 알고리즘을 통해 문서 유사도를 측정합니다.

# %%
def search_relevant_chunks(query, vector_store, top_k=3):
    """
    질문과 가장 유사한 청크를 검색

    Args:
        query: 사용자 질문
        vector_store: 벡터 저장소 (chunks, embeddings 포함)
        top_k: 반환할 상위 청크 개수

    Returns:
        list: (유사도, 청크) 튜플 리스트
    """
    # 1. 질문을 임베딩으로 변환
    query_embedding = embed_texts(
        [query],
        task_type="RETRIEVAL_QUERY"
    )[0]

    # 2. 모든 청크와의 유사도 계산
    similarities = cosine_similarity([query_embedding], vector_store['embeddings'])[0]

    # 3. 상위 k개 선택
    top_indices = np.argsort(similarities)[::-1][:top_k]

    # 4. 결과 반환
    results = []
    for idx in top_indices:
        results.append((similarities[idx], vector_store['chunks'][idx]))

    return results

# 검색 테스트
test_query = "인공지능이란 무엇인가요?"
print(f"질문: {test_query}\n")
print("검색 결과:")
relevant_chunks = search_relevant_chunks(test_query, vector_store, top_k=3)
for i, (score, chunk) in enumerate(relevant_chunks, 1):
    print(f"{i}. (유사도: {score:.4f}) {chunk}")

# %% [markdown]
# ## 6. RAG 시스템 구현
#
# 검색된 문서를 컨텍스트로 사용하여 Gemini API로 답변을 생성합니다.
# 이것이 RAG의 핵심입니다: 검색(Retrieval) + 생성(Generation)

# %%
def rag_query(query, vector_store, top_k=3):
    """
    RAG를 사용하여 쿼리에 대한 답변 생성

    Args:
        query: 사용자 질문
        vector_store: 벡터 저장소
        top_k: 검색할 상위 청크 개수

    Returns:
        tuple: (생성된 답변, 관련 청크 리스트)
    """
    # 1. 관련 문서 검색
    relevant_chunks = search_relevant_chunks(query, vector_store, top_k)

    # 2. 컨텍스트 구성
    context = "\n\n".join([
        f"[참고 문서 {i+1}] {chunk}"
        for i, (score, chunk) in enumerate(relevant_chunks)
    ])

    # 3. 프롬프트 구성
    prompt = f"""다음 문서들을 참고하여 질문에 답변해주세요. 기존에 학습된 지식 보다 참고 문서를 우선해 주세요.

참고 문서:
{context}

질문: {query}

답변:"""

    # 4. Gemini API로 답변 생성
    model = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return model.text, relevant_chunks

# %% [markdown]
# ## 7. RAG 시스템 테스트
#
# 다양한 질문으로 RAG 시스템을 테스트해봅시다.

# %%
# 테스트 쿼리들
test_queries = [
    "인공지능이란 무엇인가요?",
    "RAG는 어떻게 작동하나요?",
    "Transformer 모델에 대해 설명해주세요.",
    "벡터 데이터베이스는 무엇인가요?"
]

for query in test_queries:
    print("=" * 80)
    print(f"질문: {query}")
    print("-" * 80)

    # RAG로 답변 생성
    answer, relevant_chunks = rag_query(query, vector_store, top_k=2)

    # 검색된 문서 출력
    print("\n[참고 문서]")
    for i, (score, chunk) in enumerate(relevant_chunks, 1):
        print(f"{i}. (유사도: {score:.4f}) {chunk[:100]}...")

    # 생성된 답변 출력
    print(f"\n[답변]")
    print(answer)
    print()

# %% [markdown]
# ## 8. RAG vs 일반 LLM 비교
#
# RAG를 사용하면 LLM이 최신 정보나 특정 도메인 지식을 활용할 수 있습니다.
# 일반 LLM과 비교해봅시다.

# %%
query = "Transformer 모델에 대해 설명해주세요."

print("=" * 80)
print("일반 LLM (RAG 없이)")
print("=" * 80)
response_normal = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=query
)
print(response_normal.text)

print("\n" + "=" * 80)
print("RAG 사용")
print("=" * 80)
answer_rag, relevant_chunks = rag_query(query, vector_store, top_k=2)
print(answer_rag)

print("\n" + "=" * 80)
print("차이점:")
print("- RAG를 사용하면 제공된 문서를 근거로 더 정확하고 구체적인 답변을 생성합니다")
print("- 일반 LLM은 학습 시점의 지식만 사용하지만, RAG는 최신 문서를 활용할 수 있습니다")

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
