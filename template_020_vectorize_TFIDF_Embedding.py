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
# # 문장 벡터화: BOW, TF-IDF vs 임베딩
#
# ## 학습 목표
# - BOW (Bag of Words) 벡터화 방법 이해
# - TF-IDF 벡터화 방법 이해
# - TF-IDF 벡터 기반 유사도 계산 (cosine_similarity)
# - Sentence Transformer 임베딩 방법 이해
# - 임베딩 벡터 기반 유사도 계산
# - 두 방식의 차이점 이해 (단어 매칭 vs 의미 유사도)
#
# ## 학습 내용
# 1. BOW (Bag of Words) 벡터화
# 2. TF-IDF 벡터화 및 유사도 계산
# 3. Sentence Transformer 임베딩 및 유사도 계산
# 4. 방법론 비교: 단어 매칭 vs 의미 유사도

# %% [markdown]
# ---
# ## 1. 데이터 준비

# %%
# 금융 뉴스 예시 데이터

# %% [markdown]
# ---
# ## 2. BOW (Bag of Words) 벡터화
#
# **BOW의 특징:**
# - 단어의 순서를 무시하고 단어 빈도만 고려
# - 희소 벡터(Sparse Vector) 생성
# - 빠른 처리 속도
# - 의미적 유사도 파악 어려움

# %%
# CountVectorizer 객체 생성
# 문장들을 벡터로 변환
# 단어 목록

# %%
# DataFrame으로 시각화

# %% [markdown]
# ---
# ## 3. TF-IDF 벡터화 및 유사도 계산
#
# **TF-IDF의 특징:**
# - 단어의 중요도를 문서 내 빈도와 전체 문서에서의 희귀도를 고려
# - BOW보다 의미 있는 가중치 부여
# - 여전히 희소 벡터이지만 BOW보다 정보량이 많음

# %%
# TfidfVectorizer 객체 생성
# 문장들을 TF-IDF 벡터로 변환
# 단어 목록

# %%
# DataFrame으로 시각화

# %% [markdown]
# ### 3.1 TF-IDF 벡터 기반 유사도 계산 (cosine_similarity)

# %%
# TF-IDF 벡터 간 코사인 유사도 계산

# %%
# 유사도 행렬을 DataFrame으로 변환

# %% [markdown]
# ---
# ## 4. Sentence Transformer 임베딩 및 유사도 계산
#
# **임베딩의 특징:**
# - 문장 전체를 고정 크기의 밀집 벡터(Dense Vector)로 변환
# - 문맥과 의미를 이해하여 유사한 의미의 문장은 유사한 벡터 생성
# - 단어 순서와 문맥을 고려
# - 계산 비용이 높지만 정확도가 높음

# %%
# KURE-v1 모델 로드 (한국어 특화)
# 문장들을 벡터로 변환
# 첫 번째 문장의 임베딩 벡터 일부 확인

# %% [markdown]
# ### 4.1 임베딩 벡터 기반 유사도 계산

# %%
# cosine_similarity() 사용
# 유사도 행렬을 DataFrame으로 변환

# %% [markdown]
# ---
# ## 5. 방법론 비교: 단어 매칭 vs 의미 유사도
#
# ### 5.1 TF-IDF와 임베딩의 차이점 비교

# %%
# 테스트 케이스: 의미는 같지만 단어가 다른 문장들
    # === TF-IDF 방식 ===
    # === 임베딩 방식 ===
    # 상세 출력

# %% [markdown]
# ---
# ## 6. 학습 정리
#
# ### 6.1 벡터화 방법 요약
#
# | 방법 | 벡터화 함수 | 유사도 계산 함수 | 특징 |
# |------|------------|----------------|------|
# | BOW | `CountVectorizer().fit_transform()` | `cosine_similarity()` | 단어 빈도 기반 |
# | TF-IDF | `TfidfVectorizer().fit_transform()` | `cosine_similarity()` | 단어 중요도 기반 |
# | 임베딩 | `SentenceTransformer().encode()` | `cosine_similarity()` | 의미 기반 |
#
# ### 6.2 언제 어떤 방법을 사용할까?
#
# | 상황 | 추천 방법 | 이유 |
# |------|----------|------|
# | 빠른 키워드 추출 | BOW/TF-IDF | 빠른 처리, 해석 용이 |
# | 의미 기반 검색 | 임베딩 | 문맥 이해, 유사 의미 인식 |
# | 대량 문서 처리 (속도 중요) | TF-IDF | 빠른 처리 속도 |
# | 정확한 의미 분석 (정확도 중요) | 임베딩 | 높은 정확도 |

# %%
