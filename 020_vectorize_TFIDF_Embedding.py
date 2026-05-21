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
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 금융 뉴스 예시 데이터
finance_news = [
    "삼성전자 3분기 영업이익 급등, 사상최고 실적 기대",
    "코스피 하락세 지속, 외국인 순매도 확대에 우려",
    "반도체 시장 성장세 지속, 메모리 반도체 수요 증가",
    "삼성전자 주가 상승, 실적 호조 전망 낙관",
    "금리 인상 우려로 주식 시장 하락"
]

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
count_vectorizer = CountVectorizer()

# 문장들을 벡터로 변환
bow_features = count_vectorizer.fit_transform(finance_news)
bow_array = bow_features.toarray()

# 단어 목록
feature_names = count_vectorizer.get_feature_names_out()

print(f"\n문서 수: {bow_features.shape[0]}")
print(f"단어 수: {bow_features.shape[1]}")
print(f"\n단어 목록 (일부): {list(feature_names[:15])}...")

# %%
# DataFrame으로 시각화
df_bow = pd.DataFrame(
    bow_array,
    columns=feature_names,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[BOW 벡터 행렬]")
df_bow

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
tfidf_vectorizer = TfidfVectorizer()

# 문장들을 TF-IDF 벡터로 변환
tfidf_features = tfidf_vectorizer.fit_transform(finance_news)
tfidf_array = tfidf_features.toarray()

# 단어 목록
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

print(f"\n문서 수: {tfidf_features.shape[0]}")
print(f"단어 수: {tfidf_features.shape[1]}")

# %%
# DataFrame으로 시각화
df_tfidf = pd.DataFrame(
    tfidf_array,
    columns=tfidf_feature_names,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[TF-IDF 벡터 행렬]")
df_tfidf.round(3)

# %% [markdown]
# ### 3.1 TF-IDF 벡터 기반 유사도 계산 (cosine_similarity)

# %%
# TF-IDF 벡터 간 코사인 유사도 계산
tfidf_similarity = cosine_similarity(tfidf_features)
tfidf_similarity.shape

# %%
# 유사도 행렬을 DataFrame으로 변환
df_tfidf_sim = pd.DataFrame(
    tfidf_similarity,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[TF-IDF 기반 코사인 유사도 행렬]")
print(df_tfidf_sim.round(3))

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
print("\n임베딩 모델 로드 중...")
model = SentenceTransformer("nlpai-lab/KURE-v1")
print("✓ KURE-v1 모델 로드 완료 (한국어 특화 문장 임베딩 모델)")

# 문장들을 벡터로 변환
embeddings = model.encode(finance_news)

print(f"\n임베딩 차원: {embeddings.shape}")
print(f"  - 문장 수: {embeddings.shape[0]}")
print(f"  - 벡터 차원: {embeddings.shape[1]}")

# 첫 번째 문장의 임베딩 벡터 일부 확인
print(f"\n첫 번째 뉴스 임베딩 벡터 (처음 10개 값):")
print(embeddings[0][:10])

# %% [markdown]
# ### 4.1 임베딩 벡터 기반 유사도 계산

# %%
# cosine_similarity() 사용
embedding_similarity = cosine_similarity(embeddings)

# 유사도 행렬을 DataFrame으로 변환
df_embed_sim = pd.DataFrame(
    embedding_similarity,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))]
)
print("\n[임베딩 기반 코사인 유사도 행렬]")
df_embed_sim.round(3)

# %% [markdown]
# ---
# ## 5. 방법론 비교: 단어 매칭 vs 의미 유사도
#
# ### 5.1 TF-IDF와 임베딩의 차이점 비교

# %%
# 테스트 케이스: 의미는 같지만 단어가 다른 문장들
test_cases = [
    {
        'text1': '삼성전자 주가가 상승했습니다',
        'text2': '삼성전자 주식 가격이 올랐습니다'  # 의미는 같지만 단어가 다름
    },
    {
        'text1': '주가 상승 실적 호조 전망 낙관',
        'text2': '주식 가격 증가 실적 좋음 전망 긍정적'  # 의미 유사하지만 표현 다름
    },
    {
        'text1': '반도체 시장 성장',
        'text2': '반도체 수요 증가'  # 일부 단어 겹침
    }
]


for case in test_cases:
    text1 = case['text1']
    text2 = case['text2']

    # === TF-IDF 방식 ===
    tfidf_test = TfidfVectorizer()
    tfidf_vectors = tfidf_test.fit_transform([text1, text2])
    tfidf_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]

    # === 임베딩 방식 ===
    emb_vectors = model.encode([text1, text2])
    emb_sim = cosine_similarity([emb_vectors[0]], [emb_vectors[1]])[0][0]

    # 상세 출력
    print("-" * 80)
    print(f"  문장1: {text1}")
    print(f"  문장2: {text2}")
    print(f"\n  TF-IDF 유사도: {tfidf_sim:.4f}")
    print(f"  임베딩 유사도: {emb_sim:.4f}")
    print()

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
