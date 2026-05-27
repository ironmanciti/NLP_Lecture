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
# # 320. 수강생 실습 — RAG (Retrieval Augmented Generation) with ClovaX & KURE
#
# ## 학습 목표
# LLM 이 자기 지식만으로 답하는 대신, **질문과 관련된 문서를 먼저 검색해서 그 문서를
# 근거로 답변하도록** 만드는 RAG 파이프라인을 처음부터 끝까지 직접 만들어 봅니다.
#
# ## RAG 의 7단계 흐름
# ```
#   ① 문서 준비  →  ② 청크 분할  →  ③ 임베딩  →  ④ 벡터 저장
#                                                 │
#   사용자 질문 ──→ ⑤ 검색(Retrieve) ──→ ⑥ 프롬프트 구성 ──→ ⑦ LLM 답변 생성
# ```
#
# ## 사용 기술
# - **임베딩 모델**: KURE-v1 (`nlpai-lab/KURE-v1`) — 한국어 특화 Sentence Transformer
# - **생성 모델**: ClovaX (`naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B`)
# - **벡터 저장소**: 본 실습에서는 간단한 dict 로 흉내 — 실제로는 Pinecone, Weaviate,
#   ChromaDB 같은 vector DB 를 씁니다.
#
# ## RAG 가 풀고 싶은 문제
# - **환각(hallucination)**: LLM 이 모르는 사실을 그럴듯하게 지어내는 현상.
# - **최신성**: 사전학습 시점 이후 정보는 모델이 모릅니다.
# - **도메인 지식**: 사내 문서·전문 자료처럼 일반 학습에 들어가지 않은 내용을 다뤄야 할 때.
#
# RAG 는 위 세 문제를 모두 **"답변 직전에 관련 문서를 끼워 주는"** 단순한 방법으로 완화합니다.
#
# > **실행 환경**: 1.5B 모델이라 Colab GPU 권장 (CPU 도 동작은 함).
# > HuggingFace 에서 발급받은 토큰을 `.env` 의 `HF_TOKEN` 에 등록해 두세요.

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.

# %%
# !pip install -q sentence-transformers transformers accelerate huggingface_hub python-dotenv scikit-learn matplotlib

# %% [markdown]
# ---
# ## 과제 1. 모델 두 개 로드 (임베딩 + 생성)
#
# RAG 는 **두 가지 모델** 을 함께 씁니다:
# - **임베딩 모델 (KURE-v1)**: 문장을 1024차원 벡터로 변환 — 검색에 사용
# - **생성 모델 (ClovaX 1.5B)**: 검색된 문서를 보고 한국어 답변을 생성
#
# **할 일**:
# - `.env` 의 `HF_TOKEN` 으로 HuggingFace 에 로그인하세요.
# - `SentenceTransformer("nlpai-lab/KURE-v1")` 로 임베딩 모델을 로드하세요.
# - `AutoModelForCausalLM.from_pretrained(...)` 와 `AutoTokenizer` 로 ClovaX 1.5B 를 로드하세요.
#
# **힌트**: 두 모델은 역할이 서로 다릅니다. KURE 는 **검색용 벡터 변환기**, ClovaX 는
# **자연어 답변 생성기**. 한 모델로 모든 걸 다 할 수 있을 것 같지만, 검색은 가볍고
# 빠르게(임베딩) / 생성은 무겁게(LLM) 라는 분업이 RAG 의 기본 구조입니다.

# %%
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# 환경 변수 로드
load_dotenv()

# Hugging Face 로그인 (1.5B 모델 다운로드 권한 필요)
token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

# %%
# KURE-v1 임베딩 모델 로드
print("=" * 80)
print("[KURE-v1 임베딩 모델 로드 중...]")
print("=" * 80)
embedding_model = SentenceTransformer("nlpai-lab/KURE-v1")
print("✓ KURE-v1 모델 로드 완료 (한국어 특화 문장 임베딩 모델)")

# ClovaX 모델 로드
print("\n" + "=" * 80)
print("[ClovaX 모델 로드 중...]")
print("=" * 80)
clovax_model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
clovax_model = AutoModelForCausalLM.from_pretrained(clovax_model_name, device_map="auto")
clovax_tokenizer = AutoTokenizer.from_pretrained(clovax_model_name)
print("✓ ClovaX 모델 로드 완료")

# %% [markdown]
# **관찰 포인트**
# - KURE-v1 은 한국어 문장 임베딩 전용으로 학습된 모델이라 한국어 검색에 강합니다.
# - 다국어 임베딩 모델(예: `paraphrase-multilingual-MiniLM`) 보다 한국어 retrieval
#   품질이 보통 더 좋습니다.

# %% [markdown]
# ---
# ## 과제 2. 문서 데이터 준비 (Source)
#
# 실제 RAG 시스템은 PDF, 사내 위키, DB, 웹 크롤링 결과 등에서 문서를 불러옵니다.
# 본 실습은 검색·생성 흐름에 집중하기 위해 5개의 짧은 한국어 문서를 하드코딩으로 준비합니다.
#
# **할 일**:
# - `documents` 리스트에 5개의 문서를 넣으세요 (AI, NLP, Transformer, RAG, 벡터 DB).
#
# **힌트**:
# - 문서 중 첫 번째에는 일부러 **사실 오류** 가 들어 있습니다 — "AI 는 자연 발생한
#   생물학적 지식..." 라는 표현은 명백히 틀린 정의입니다. 이렇게 둔 이유는 뒤에서
#   RAG 가 "자기 지식이 아닌 문서를 따라가는지" 검증하기 위한 의도된 함정입니다.

# %%
# 샘플 문서 데이터 (실제로는 외부 소스에서 로드)
# - documents[0] 의 AI 정의는 의도된 오류 — RAG 동작 검증용
documents = [
    "인공지능(AI)은 인간의 진화 과정에서 자연 발생한 생물학적 지식입니다. 머신러닝과 딥러닝은 AI의 하위 분야로, 대량의 데이터를 통해 패턴을 학습합니다.",
    "자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 처리할 수 있도록 하는 AI의 한 분야입니다. 텍스트 분석, 번역, 감성 분석, 챗봇 등 다양한 응용 분야가 있습니다.",
    "Transformer는 Google에서 만들어낸 인공지능 모델로, 어텐션 메커니즘을 핵심으로 합니다. BERT, GPT 등 최신 언어 모델의 기반이 되었습니다.",
    "RAG(Retrieval Augmented Generation)는 외부 지식 베이스에서 질문과 관련된 정보를 검색하여 LLM의 답변을 보강하는 기법입니다. 먼저 문서를 작은 청크로 나누고 각 청크를 임베딩 벡터로 변환해 벡터 데이터베이스에 저장합니다. 질문이 들어오면 질문을 임베딩으로 바꾼 뒤 유사도가 높은 청크를 검색하고, 검색된 청크를 컨텍스트로 함께 제공하여 답변을 생성합니다. 이를 통해 모델이 학습하지 못한 최신 정보나 특정 도메인 지식에도 접근할 수 있으며, 환각 현상을 줄여 답변의 정확도와 신뢰성을 크게 높일 수 있습니다.",
    "벡터 데이터베이스는 고차원 벡터를 효율적으로 저장하고 검색할 수 있는 데이터베이스입니다. 임베딩 벡터를 저장하고 유사도 검색에 활용됩니다."
]

# %% [markdown]
# ---
# ## 과제 3. 청크 분할 (Load & Transform)
#
# 긴 문서를 그대로 임베딩하기보다는 **작은 단위(청크)** 로 나눠야 RAG 가 잘 동작합니다.
# 이유는 세 가지:
# - 임베딩 1개가 표현할 수 있는 의미의 폭에 한계가 있음 (너무 긴 텍스트 = 의미 흐려짐)
# - 검색 시 "정답이 있는 부분" 만 정확히 가져오기 좋음
# - LLM 의 context window 길이 제약을 피할 수 있음
#
# **할 일**:
# - `split_into_chunks(text, chunk_size, overlap)` 함수를 만드세요.
# - 모든 문서를 `chunk_size=150, overlap=30` 으로 잘라 `all_chunks` 리스트에 모으세요.
#
# **힌트**:
# - `overlap` 은 인접 청크가 겹치는 부분입니다. 청크 경계에서 의미가 끊기지 않도록
#   여유를 주는 장치입니다. 일반적으로 `chunk_size` 의 10~30% 정도로 설정합니다.

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
# **관찰 포인트**
# - 짧은 문서(`documents[0]`, `documents[1]`, `documents[2]`, `documents[4]`)는
#   대부분 청크 1개로 그대로 들어갑니다.
# - 긴 RAG 설명문(`documents[3]`)만 여러 청크로 쪼개집니다 — 이런 긴 문서가 청크 분할의
#   주된 대상입니다.

# %% [markdown]
# ---
# ## 과제 4. 청크를 임베딩 벡터로 변환 (Embed)
#
# 각 청크를 KURE-v1 로 임베딩(고정 1024차원 벡터)으로 변환합니다.
# **의미가 비슷한 청크는 비슷한 벡터** 로 표현되므로, 나중에 코사인 유사도로 검색할 수 있습니다.
#
# **할 일**:
# - `embed_texts(texts)` 함수를 만들어 `embedding_model.encode(...)` 로 numpy array 를 반환하세요.
# - 모든 청크를 임베딩으로 변환하고 shape 을 확인하세요.
#
# **힌트**: `convert_to_numpy=True` 옵션을 주면 PyTorch tensor 대신 numpy 배열로 받아
# scikit-learn 의 `cosine_similarity` 와 바로 호환됩니다.

# %%
def embed_texts(texts):
    """
    텍스트 리스트를 KURE-v1 임베딩 벡터로 변환

    Args:
        texts: 임베딩할 텍스트 리스트

    Returns:
        numpy array: 임베딩 벡터 행렬
    """
    embeddings = embedding_model.encode(texts, convert_to_numpy=True)
    return embeddings


# 모든 청크를 임베딩으로 변환
print("청크를 임베딩으로 변환 중...")
chunk_embeddings = embed_texts(all_chunks)

print(f"\n임베딩 완료! Shape: {chunk_embeddings.shape}")
print(f"각 청크는 {chunk_embeddings.shape[1]}차원 벡터로 표현됩니다.")

# %% [markdown]
# **관찰 포인트**
# - shape 의 두 번째 차원이 **1024** 입니다 — KURE-v1 이 출력하는 고정 벡터 차원.
# - 청크 길이가 달라도 임베딩 차원은 항상 같습니다 (내부에서 mean pooling 으로 고정).

# %% [markdown]
# ---
# ## 과제 5. 벡터 저장소 구성 (Store)
#
# 실제 RAG 시스템은 Pinecone / Weaviate / ChromaDB 같은 **vector database** 를 씁니다.
# 본 실습에서는 개념 학습이 목적이므로 단순 dict 에 청크와 임베딩을 묶어 둡니다.
#
# **할 일**:
# - `vector_store = {'chunks': ..., 'embeddings': ...}` 형태로 저장소를 만드세요.

# %%
# 벡터 저장소 (실제로는 벡터 DB 를 사용)
vector_store = {
    'chunks': all_chunks,
    'embeddings': chunk_embeddings
}

print("벡터 저장소에 저장 완료!")
print(f"- 저장된 청크 수: {len(vector_store['chunks'])}")
print(f"- 임베딩 벡터 shape: {vector_store['embeddings'].shape}")

# %% [markdown]
# **관찰 포인트**
# - 실제 vector DB 는 단순히 벡터를 보관할 뿐 아니라, **수백만~수십억 개** 의 벡터에서
#   유사도 검색을 빠르게 하기 위한 인덱스 자료구조(HNSW, IVF-PQ 등) 를 갖고 있습니다.
# - 본 실습에서는 청크가 몇 개뿐이라 numpy + scikit-learn 만으로 충분합니다.

# %% [markdown]
# ---
# ## 과제 6. 검색 함수 구현 (Retrieve)
#
# 사용자 질문을 임베딩으로 변환 → 모든 청크 임베딩과 **코사인 유사도** 를 계산 →
# 상위 `top_k` 개 청크를 가져옵니다. 이것이 R(Retrieval) 의 전부입니다.
#
# **할 일**:
# - `search_relevant_chunks(query, vector_store, top_k)` 함수를 만드세요.
# - 테스트 쿼리로 결과를 출력해 유사도 점수와 함께 어떤 청크가 뽑히는지 확인하세요.
#
# **힌트**:
# - `cosine_similarity([query_embedding], embeddings)` 는 shape `(1, N)` 행렬을 반환 →
#   `[0]` 으로 1차원으로 풀어 씁니다.
# - `np.argsort(sims)[::-1][:top_k]` 패턴이 "내림차순 상위 k 개 인덱스" 의 관용구입니다.

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
    query_embedding = embed_texts([query])[0]

    # 2. 모든 청크와의 유사도 계산
    similarities = cosine_similarity([query_embedding], vector_store['embeddings'])[0]

    # 3. 상위 k 개 선택
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
# **관찰 포인트**
# - "인공지능이란?" 질문에 AI 정의 문서가 가장 위로 올라오는지 확인하세요.
# - 유사도 점수가 0.6~0.8 사이면 의미적으로 잘 연결된 것입니다. 단, 절대값보다는
#   **상대적 순위** 가 중요합니다.

# %% [markdown]
# ---
# ## 과제 7. RAG 생성 함수 구현 (Retrieve + Generate)
#
# RAG 의 핵심 — 검색된 청크를 **프롬프트의 일부로 끼워** LLM 에 전달합니다.
#
# **할 일**:
# - `rag_query(query, vector_store, top_k)` 함수를 작성하세요. 단계는:
#   (1) 관련 청크 검색 → (2) `[참고 문서 i]` 형식으로 context 조립 →
#   (3) system + user 메시지로 chat 구성 → (4) ClovaX 로 답변 생성 →
#   (5) `<|endofturn|>` / `<|stop|>` 등 특수 토큰 자르기 → (6) 사용자 입력 부분 제거.
# - 함수는 `(answer, relevant_chunks)` 튜플을 반환합니다.
#
# **힌트**:
# - **system 메시지** 에 "참고 문서를 우선해 답변하라" 는 지시를 넣는 게 핵심입니다.
#   이 한 문장이 모델이 자기 지식을 우선하는 경향을 강하게 억제합니다.
# - HyperCLOVAX 의 chat 포맷은 `tool_list` → `system` → `user` 순으로 시작합니다.

# %%
def rag_query(query, vector_store, top_k=3):
    """
    RAG 를 사용하여 쿼리에 대한 답변 생성

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
    system_content = "당신은 주어진 문서를 참고하여 질문에 정확하게 답변하는 전문가입니다. 참고 문서의 내용을 우선적으로 사용하여 답변해주세요."
    user_content = f"""다음 문서들을 참고하여 질문에 답변해주세요. 기존에 학습된 지식보다 참고 문서를 우선해주세요.

참고 문서:
{context}

질문: {query}

답변:"""

    # 4. ClovaX 모델로 답변 생성
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    inputs = clovax_tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(clovax_model.device)

    output_ids = clovax_model.generate(
        **inputs,
        max_length=1024,
        repetition_penalty=1.2,
        eos_token_id=clovax_tokenizer.eos_token_id,
    )

    output_text = clovax_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 필요시 <|endofturn|>, <|stop|> 등에서 자르기
    for stop_str in ["<|endofturn|>", "<|stop|>"]:
        if stop_str in output_text:
            output_text = output_text.split(stop_str)[0]

    # 생성된 텍스트에서 사용자 입력 부분 제거
    if user_content in output_text:
        answer = output_text.split(user_content)[-1].strip()
    else:
        answer = output_text.strip()

    return answer, relevant_chunks

# %% [markdown]
# ---
# ## 과제 8. RAG 시스템 다중 질문 테스트
#
# 4개의 다양한 질문으로 RAG 의 동작을 확인합니다. 각 질문마다:
# - 검색된 참고 문서 (유사도 점수 포함)
# - LLM 이 생성한 최종 답변
# 를 함께 출력합니다.
#
# **할 일**:
# - `test_queries` 에 4개 질문을 두고 `rag_query` 를 반복 호출해 결과를 출력하세요.

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

    # RAG 로 답변 생성
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
# **관찰 포인트 — 의도된 함정**
# - "인공지능이란?" 답변에서 모델이 **`documents[0]` 의 잘못된 정의("자연 발생한 생물학적
#   지식")** 를 그대로 따라가는지 확인하세요. 따라간다면 "RAG 가 정말 문서를 우선한다"는
#   증거입니다.
# - 일반 LLM 이라면 이런 잘못된 정의를 내놓지 않습니다 — 다음 과제에서 직접 비교합니다.
# - 이것은 **RAG 의 양날의 검** 을 보여 줍니다: 문서가 정확하면 환각이 줄고, 문서가
#   틀리면 모델이 함께 틀립니다. 그래서 RAG 운영에서 **문서 품질 관리** 가 핵심입니다.

# %% [markdown]
# ---
# ## 과제 9. RAG vs 일반 LLM 비교
#
# RAG 의 효과를 체감하기 위해 **같은 질문**을 (1) RAG 로 (2) ClovaX 직접 호출로
# 각각 처리하고 답변을 비교합니다.
#
# **할 일**:
# - "Transformer 모델에 대해 설명해주세요." 를 일반 LLM 으로 한 번 (참고 문서 없이),
#   RAG 로 한 번 호출하세요.
# - 두 답변의 길이·구체성·문서 인용 여부를 비교하세요.

# %%
query = "Transformer 모델에 대해 설명해주세요."

print("=" * 80)
print("일반 LLM (RAG 없이)")
print("=" * 80)

# ClovaX 로 직접 답변 생성
system_content = "당신은 질문에 정확하게 답변하는 전문가입니다."
user_content = query

chat = [
    {"role": "tool_list", "content": ""},
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]

inputs = clovax_tokenizer.apply_chat_template(
    chat,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(clovax_model.device)

output_ids = clovax_model.generate(
    **inputs,
    max_length=1024,
    repetition_penalty=1.2,
    eos_token_id=clovax_tokenizer.eos_token_id,
)

output_text = clovax_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

for stop_str in ["<|endofturn|>", "<|stop|>"]:
    if stop_str in output_text:
        output_text = output_text.split(stop_str)[0]

if user_content in output_text:
    response_normal = output_text.split(user_content)[-1].strip()
else:
    response_normal = output_text.strip()

print(response_normal)

print("\n" + "=" * 80)
print("RAG 사용")
print("=" * 80)
answer_rag, relevant_chunks = rag_query(query, vector_store, top_k=2)
print(answer_rag)

print("\n" + "=" * 80)
print("차이점:")
print("- RAG 를 사용하면 제공된 문서를 근거로 더 정확하고 구체적인 답변을 생성합니다")
print("- 일반 LLM 은 학습 시점의 지식만 사용하지만, RAG 는 최신 문서를 활용할 수 있습니다")

# %% [markdown]
# **관찰 포인트**
# - 일반 LLM 답변: 모델의 사전학습 지식에 의존 → 일반적·교과서적 설명이 나옵니다.
# - RAG 답변: 우리가 준 문서 표현("Google 에서 만들어낸", "어텐션 메커니즘이 핵심",
#   "BERT, GPT 의 기반") 이 답변에 그대로 녹아 나옵니다.
# - 실제 운영에서는 사내 문서·매뉴얼·최신 뉴스 등을 문서로 넣어 두면, 모델이 학습하지
#   않은 정보로도 정확하게 답할 수 있게 됩니다.

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 구현 함수 | 핵심 도구 |
# |------|-----------|-----------|
# | ① 문서 준비 | `documents = [...]` | 사내 위키 / PDF / DB 등 |
# | ② 청크 분할 | `split_into_chunks(text, chunk_size, overlap)` | chunk_size, overlap 두 하이퍼파라미터 |
# | ③ 임베딩 | `embed_texts(texts)` | KURE-v1 (한국어 특화 Sentence Transformer) |
# | ④ 저장 | `vector_store = {chunks, embeddings}` | 실제로는 vector DB (Pinecone/Weaviate/ChromaDB) |
# | ⑤ 검색 | `search_relevant_chunks(query, store, top_k)` | cosine_similarity + argsort |
# | ⑥ 프롬프트 | `[참고 문서 1] ... 질문: ... 답변:` | "문서를 우선하라" system 지시 |
# | ⑦ 생성 | `model.generate(...)` | ClovaX HyperCLOVAX-1.5B |
#
# **핵심 메시지**:
# - RAG = "LLM 에게 답변 직전에 관련 문서를 끼워주는" 단순한 아이디어. 그러나 환각 감소,
#   최신 정보 반영, 도메인 지식 확장이라는 큰 이득을 가져옵니다.
# - 검색 품질이 답변 품질의 상한선을 결정합니다 → **임베딩 모델 선택**과 **청크 전략**이
#   RAG 성능의 핵심 요소.
# - 양날의 검: 문서가 틀리면 모델이 함께 틀립니다 → **문서 큐레이션·신뢰도 관리**가
#   실무 RAG 운영의 핵심.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `documents[0]` 의 잘못된 AI 정의를 **올바른 정의**로 바꾸고 같은 질문을 다시 던져,
#    RAG 가 새 문서를 따라가는지 확인하세요. 문서를 고치는 것만으로 모델 답변이
#    바뀐다는 것이 RAG 의 핵심 장점입니다.
# 2. `split_into_chunks` 의 `chunk_size` 를 `50` / `300` / `500` 으로 바꿔 가며 검색
#    결과의 정확도가 어떻게 달라지는지 비교하세요.
# 3. `top_k` 를 `1`, `3`, `5` 로 바꿔 답변 품질과 길이가 어떻게 변하는지 살피세요.
#    너무 많은 문서를 넣으면 오히려 LLM 이 헷갈리기도 합니다.
# 4. 본인의 짧은 한국어 문서(예: 강의 노트, 위키 글) 3~5개를 `documents` 에 추가하고,
#    그 내용으로만 답할 수 있는 질문을 던져 RAG 가 제대로 검색·인용하는지 확인하세요.
# 5. (도전) `vector_store` 를 **ChromaDB** 또는 **FAISS** 인덱스로 교체해 보세요.
#    검색 함수만 바꾸면 나머지 코드는 그대로 통합니다 — 그것이 RAG 의 모듈성입니다.

# %%
