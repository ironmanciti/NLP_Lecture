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
# # 030. 자율 실습 — 문장에서 임베딩까지
#
# 010 (토큰화) 과 020 (TF-IDF / 임베딩 유사도) 에서 배운 흐름을
# **KURE-v1 한 모델 안에서** 직접 손으로 따라가 봅니다.
#
# ```
#   문장 (str)
#     │  tokenize
#     ▼
#   토큰 / ID 시퀀스
#     │  encode
#     ▼
#   1024-dim 임베딩
#     │  cosine
#     ▼
#   유사도 행렬
# ```
#
# ## 학습 목표
# - SentenceTransformer 모델 내부의 **tokenizer** 와 **encoder** 를 분리해서 들여다보기
# - `tokenize` / `encode` / `decode` 의 차이 이해
# - 문장마다 토큰 수가 달라도 임베딩은 **고정 1024차원** 으로 나오는 이유 (mean pooling) 확인
# - 임베딩 벡터 간 코사인 유사도로 의미 유사도 측정
#
# ## 과제 단계
# | # | 단계 | 핵심 함수 |
# |---|------|-----------|
# | 1 | 모델 로드 | `SentenceTransformer('nlpai-lab/KURE-v1')` |
# | 2 | 토큰 문자열 | `tokenizer.tokenize(s)` → `['▁삼성', '전자', ...]` |
# | 3 | 토큰 ID | `tokenizer.encode(s)` → `[0, ..., 2]` (`<s>`, `</s>`) |
# | 4 | 원문 복원 | `tokenizer.decode(ids)` |
# | 5 | 문장 임베딩 | `model.encode(sentences)` → shape `(4, 1024)` |
# | 6 | 길이 비교 | 가변 토큰 수 → 고정 1024차원 (mean pooling) |
# | 7 | 유사도 계산 | `cosine_similarity(embeddings)` |

# %% [markdown]
# ---
# ## 0. 환경 준비
#
# Colab 에서는 아래 설치 명령을 먼저 실행하세요.

# %%

# %%
# 실습용 한국어 문장 (의미 유사 / 비유사 쌍이 섞이도록 구성)

# %% [markdown]
# ---
# ## 과제 1. 모델 로드
#
# `SentenceTransformer` 는 내부적으로 **tokenizer + transformer + pooling** 을
# 하나의 파이프라인으로 묶은 wrapper 입니다.
# 단일 모델 안에서 토큰화부터 임베딩까지 한 번에 처리할 수 있습니다.
#
# **할 일**
# - `nlpai-lab/KURE-v1` 모델을 로드하세요.
# - 모델 내부의 tokenizer 객체를 `tokenizer` 변수에 꺼내 두세요.
#
# **힌트**: `model.tokenizer` 로 내부 HuggingFace tokenizer 에 접근할 수 있습니다.

# %%

# %% [markdown]
# ---
# ## 과제 2. 토큰 문자열 (`tokenize`)
#
# 문장을 **사람이 읽을 수 있는 sub-word 토큰 문자열** 로 쪼갭니다.
# KURE-v1 은 SentencePiece 기반이라 단어 앞에 `▁` (U+2581) 가 붙어
# **공백을 토큰의 일부로 표현** 합니다.
#
# **할 일**
# - 각 문장을 `tokenizer.tokenize()` 로 토큰 리스트로 변환하세요.

# %%

# %% [markdown]
# **관찰 포인트**
# - 단어 시작에 `▁` 가 붙어 띄어쓰기 위치를 보존합니다.
# - "삼성전자" 가 한 토큰일 수도, "▁삼성" + "전자" 로 나뉠 수도 있습니다 — vocab 빈도에 의존.
# - 문장마다 토큰 개수가 **다릅니다** (가변 길이).

# %% [markdown]
# ---
# ## 과제 3. 토큰 ID (`encode`)
#
# 모델은 문자열이 아니라 **정수 ID** 로 동작합니다.
# `encode()` 는 토큰을 ID 로 바꾸고, 추가로 **특수 토큰** 을 양 끝에 붙입니다.
#
# - `<s>` (id `0`) : sequence start
# - `</s>` (id `2`) : sequence end
#
# **할 일**
# - 각 문장을 `tokenizer.encode()` 로 정수 시퀀스로 변환하세요.
# - 첫·끝 토큰이 `0`, `2` 인지 확인하세요.

# %%

# %% [markdown]
# **관찰 포인트**
# - 모든 시퀀스가 `0 (<s>)` 으로 시작하고 `2 (</s>)` 로 끝납니다.
# - `tokenize()` 결과보다 ID 시퀀스가 **2개 더 깁니다** (특수 토큰만큼).

# %% [markdown]
# ---
# ## 과제 4. 원문 복원 (`decode`)
#
# `decode()` 는 정수 ID 시퀀스를 다시 문자열로 되돌립니다.
# 토큰화/인코딩이 가역적인지 확인할 수 있습니다.
#
# **할 일**
# - 위에서 만든 ID 를 다시 `tokenizer.decode()` 로 문자열로 복원하세요.
# - `skip_special_tokens=True` 옵션의 효과를 비교해 보세요.

# %%

# %% [markdown]
# **관찰 포인트**
# - `skip_special_tokens=False` 면 `<s>`, `</s>` 가 그대로 보입니다.
# - SentencePiece 의 `▁` 는 decode 시 자동으로 공백으로 복원됩니다.

# %% [markdown]
# ---
# ## 과제 5. 문장 임베딩 (`model.encode`)
#
# 지금까지는 **tokenizer** 만 다뤘습니다. 이제 **모델 전체** 를 호출해
# 문장 단위의 의미 벡터를 얻습니다.
#
# **할 일**
# - `model.encode(sentences)` 로 임베딩을 생성하세요.
# - 결과 shape 이 `(문장 수, 1024)` 인지 확인하세요.

# %%

# %% [markdown]
# ---
# ## 과제 6. 길이 비교 — 가변 토큰 수 vs 고정 1024차원
#
# 핵심 질문: **토큰 개수가 문장마다 다른데 임베딩은 왜 모두 1024차원일까?**
#
# 답: SentenceTransformer 는 transformer 가 만든 토큰별 벡터를
# **mean pooling** (또는 [CLS] pooling) 으로 평균 내어 한 개의 고정 차원 벡터로 압축합니다.
#
# **할 일**
# - 각 문장의 토큰 수와 임베딩 차원을 표로 정리해 확인하세요.

# %%

# %% [markdown]
# **관찰 포인트**
# - 토큰 수: 문장마다 다름 (예: 7, 9, 8, 6 …).
# - 임베딩 차원: 모두 **1024 로 동일** — 이후 단계에서 행렬 연산이 가능해지는 이유.
# - 이 압축 단계가 mean pooling 이며, `SentenceTransformer` 내부의 마지막 모듈입니다.

# %% [markdown]
# ---
# ## 과제 7. 유사도 계산
#
# 1024차원으로 정렬된 임베딩 사이의 **코사인 유사도** 로 의미 거리를 측정합니다.
#
# **할 일**
# - `cosine_similarity(embeddings)` 로 유사도 행렬을 만드세요.
# - DataFrame 으로 보기 좋게 출력하세요.
# - 의미가 거의 같은 **문장1 ↔ 문장2** 의 유사도가 가장 높게 나오는지 확인하세요.

# %%

# %% [markdown]
# ### 가장 비슷한 문장 쌍 찾기

# %%

# %% [markdown]
# **관찰 포인트**
# - "삼성전자 주가가 상승했다" ↔ "삼성전자 주식 가격이 올랐다" 는 단어가 거의 겹치지 않는데도
#   유사도가 매우 높게 나옵니다 → **의미 기반 임베딩** 의 효과.
# - 020 의 TF-IDF 로 같은 문장을 비교했을 때와 점수를 비교해 보세요.

# %% [markdown]
# ---
# ## 정리
#
# | 단계 | 입력 | 출력 | 핵심 함수 |
# |------|------|------|-----------|
# | 1. 모델 로드 | 모델 ID | tokenizer + encoder | `SentenceTransformer(...)` |
# | 2. 토큰 문자열 | 문장 (str) | 토큰 리스트 | `tokenizer.tokenize` |
# | 3. 토큰 ID | 문장 (str) | 정수 시퀀스 | `tokenizer.encode` |
# | 4. 원문 복원 | ID 시퀀스 | 문장 (str) | `tokenizer.decode` |
# | 5. 문장 임베딩 | 문장 리스트 | (N, 1024) 행렬 | `model.encode` |
# | 6. 길이 비교 | 가변 토큰 수 | 고정 1024차원 | mean pooling (내부) |
# | 7. 유사도 | 임베딩 행렬 | (N, N) 유사도 | `cosine_similarity` |
#
# **핵심 통찰**
# - `tokenize` 는 "사람이 보는 토큰", `encode` 는 "모델이 보는 정수 ID" — 둘은 같은 작업의 다른 표현.
# - 토큰 수가 가변이어도 mean pooling 덕분에 모든 문장이 같은 1024차원에 놓입니다.
# - 그래서 행렬 연산 (코사인 유사도, 클러스터링, 검색) 이 그대로 가능합니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택)
#
# 1. `sentences` 에 **본인이 만든 문장 3개** 를 추가하고, 의미가 가까운 쌍을 예측한 뒤
#    유사도 행렬로 확인하세요.
# 2. `tokenizer.tokenize` 결과에서 `▁` 가 붙은 토큰과 안 붙은 토큰의 차이를 정리하세요.
# 3. 020 의 TF-IDF 방식으로 같은 `sentences` 의 유사도를 구해 표를 나란히 두고 비교하세요.
#    어떤 쌍에서 점수 차이가 크게 벌어지는지, 왜 그런지 한 줄로 설명해 보세요.
# 4. `model.encode(sentences, normalize_embeddings=True)` 로 정규화한 뒤
#    `embeddings @ embeddings.T` 로도 같은 유사도가 나오는지 확인하세요.

# %%
