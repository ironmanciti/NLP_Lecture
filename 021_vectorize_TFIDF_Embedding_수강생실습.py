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
# # 021. 수강생 실습 - 문장 벡터화 3종 비교 (BOW / TF-IDF / 임베딩)
#
# ## 학습 목표
# 컴퓨터는 문장을 그대로 이해하지 못하므로 **숫자 벡터**로 바꿔야 합니다.
# 같은 문장들을 3가지 방식으로 벡터화하고, 그 벡터로 **문장 간 유사도**를 계산하며
# 각 방식의 원리와 한계를 직접 관찰합니다.
#
# 특히 마지막 과제에서 **"단어는 다르지만 의미는 같은 문장"** 을 다룰 때
# TF-IDF 와 임베딩이 어떻게 갈리는지에 주목하세요.
#
# | 벡터화 | 방식 | 특징 |
# |--------|------|------|
# | BOW | 단어 빈도수 | 단어 순서 무시, 희소 벡터 |
# | TF-IDF | 단어 중요도 가중치 | 흔한 단어는 낮게, 희귀 단어는 높게 |
# | 임베딩 | 의미 기반 밀집 벡터 | 단어가 달라도 의미가 비슷하면 가까움 |

# %%
# 실습에 필요한 패키지 설치 (Colab 기준 — 로컬에 이미 설치돼 있으면 생략 가능)
# !pip install -q scikit-learn sentence-transformers

# %%
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# 실습용 금융 뉴스 문장 — 모든 과제에서 공통으로 사용합니다.
finance_news = [
    "삼성전자 3분기 영업이익 급등, 사상최고 실적 기대",
    "코스피 하락세 지속, 외국인 순매도 확대에 우려",
    "반도체 시장 성장세 지속, 메모리 반도체 수요 증가",
    "삼성전자 주가 상승, 실적 호조 전망 낙관",
    "금리 인상 우려로 주식 시장 하락",
]

# %% [markdown]
# ---
# ## 과제 1. BOW (Bag of Words) 벡터화하기
#
# BOW 는 **단어의 등장 횟수**만 세어 문장을 벡터로 만듭니다. 단어 순서는 무시합니다.
#
# **할 일**:
# - `CountVectorizer` 객체를 만들고 `fit_transform(finance_news)` 로 문장을 벡터화하세요.
# - `get_feature_names_out()` 으로 단어 목록을 얻으세요.
# - 결과를 DataFrame 으로 만들어 어떤 뉴스에 어떤 단어가 몇 번 나왔는지 확인하세요.

# %%
# CountVectorizer 객체 생성
count_vectorizer = CountVectorizer()

# 문장들을 단어 빈도 벡터로 변환
bow_features = count_vectorizer.fit_transform(finance_news)
bow_array = bow_features.toarray()

# 단어 목록 추출
feature_names = count_vectorizer.get_feature_names_out()

print(f"문서 수: {bow_features.shape[0]}, 단어 수: {bow_features.shape[1]}")

# DataFrame 으로 시각화
df_bow = pd.DataFrame(
    bow_array,
    columns=feature_names,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
)
print("\n[BOW 벡터 행렬]")
df_bow

# %% [markdown]
# **관찰 포인트**
# - 각 열은 하나의 단어, 각 값은 그 단어가 해당 뉴스에 등장한 **횟수**입니다.
# - 행렬 대부분의 값이 **0** 입니다 → 이런 벡터를 **희소 벡터(sparse vector)** 라고 합니다.
# - 단어 순서를 버리므로 "주가 상승" 과 "상승 주가" 는 똑같은 벡터가 됩니다.

# %% [markdown]
# ---
# ## 과제 2. TF-IDF 벡터화하기
#
# TF-IDF 는 단순 빈도 대신 **단어의 중요도**를 가중치로 줍니다.
# 한 문서에 자주 나오면서(TF↑) 전체 문서에서는 드문(IDF↑) 단어일수록 높은 점수를 받습니다.
#
# **할 일**:
# - `TfidfVectorizer` 객체를 만들고 `fit_transform(finance_news)` 로 벡터화하세요.
# - 결과를 DataFrame 으로 만들어 과제 1의 BOW 행렬과 값이 어떻게 다른지 비교하세요.
#
# **힌트**: BOW 는 정수(빈도), TF-IDF 는 0~1 사이 실수(가중치)가 나옵니다.

# %%
# TfidfVectorizer 객체 생성
tfidf_vectorizer = TfidfVectorizer()

# 문장들을 TF-IDF 벡터로 변환
tfidf_features = tfidf_vectorizer.fit_transform(finance_news)
tfidf_array = tfidf_features.toarray()

tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
print(f"문서 수: {tfidf_features.shape[0]}, 단어 수: {tfidf_features.shape[1]}")

# DataFrame 으로 시각화
df_tfidf = pd.DataFrame(
    tfidf_array,
    columns=tfidf_feature_names,
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
)
print("\n[TF-IDF 벡터 행렬]")
df_tfidf.round(3)

# %% [markdown]
# **관찰 포인트**
# - BOW 의 정수 빈도가 TF-IDF 에서는 **0~1 사이 실수 가중치**로 바뀌었습니다.
# - 여러 뉴스에 공통으로 등장하는 흔한 단어는 가중치가 낮고,
#   특정 뉴스에만 나오는 단어는 가중치가 높습니다.
# - 하지만 여전히 0이 많은 **희소 벡터**이고, 단어가 겹치지 않으면 서로를 비교할 수 없습니다.

# %% [markdown]
# ---
# ## 과제 3. TF-IDF 벡터로 문장 간 유사도 계산하기
#
# 벡터끼리 얼마나 비슷한지는 **코사인 유사도(cosine similarity)** 로 잽니다 (1에 가까울수록 유사).
#
# **할 일**:
# - `cosine_similarity(tfidf_features)` 로 5×5 유사도 행렬을 계산하세요.
# - DataFrame 으로 보기 좋게 출력하고, 어떤 뉴스끼리 유사도가 높은지 확인하세요.

# %%
# TF-IDF 벡터 간 코사인 유사도 계산
tfidf_similarity = cosine_similarity(tfidf_features)

df_tfidf_sim = pd.DataFrame(
    tfidf_similarity.round(3),
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))],
)
print("[TF-IDF 기반 코사인 유사도 행렬]")
print(df_tfidf_sim)

# %% [markdown]
# **관찰 포인트**
# - 뉴스1 "삼성전자 ... 실적 ..." 과 뉴스4 "삼성전자 ... 실적 ..." 은
#   **'삼성전자', '실적'** 단어를 공유하므로 유사도가 나타납니다.
# - 반대로 공유하는 단어가 하나도 없는 뉴스 쌍은 유사도가 **0** 입니다.
# - → TF-IDF 유사도는 결국 **"같은 단어가 겹치는가"** 에 의존합니다.

# %% [markdown]
# ---
# ## 과제 4. Sentence Transformer 임베딩 만들기
#
# 임베딩은 문장 전체를 **의미를 담은 고정 크기 밀집 벡터(dense vector)** 로 바꿉니다.
# 여기서는 한국어 특화 모델 **KURE-v1** 을 사용합니다.
#
# **할 일**:
# - `SentenceTransformer("nlpai-lab/KURE-v1")` 로 모델을 로드하세요.
# - `model.encode(finance_news)` 로 문장 임베딩을 얻고 결과 shape 을 확인하세요.

# %%
# KURE-v1 임베딩 모델 로드 (한국어 특화)
print("임베딩 모델 로드 중...")
model = SentenceTransformer("nlpai-lab/KURE-v1")
print("KURE-v1 모델 로드 완료")

# 문장들을 임베딩 벡터로 변환
embeddings = model.encode(finance_news)
print(f"\n임베딩 shape: {embeddings.shape}   # (문장 수, 벡터 차원)")

# 첫 번째 뉴스의 임베딩 벡터 일부 확인
print(f"\n첫 번째 뉴스 임베딩 (앞 10개 값):")
print(np.round(embeddings[0][:10], 4))

# %% [markdown]
# **관찰 포인트**
# - 5개 문장이 각각 **1024차원 실수 벡터**로 변환되었습니다.
# - BOW/TF-IDF 와 달리 값이 거의 0이 아닙니다 → **밀집 벡터(dense vector)**.
# - 벡터 차원은 단어 수와 무관하게 **항상 1024** 로 고정됩니다.

# %% [markdown]
# ---
# ## 과제 5. 임베딩 벡터로 문장 간 유사도 계산하기
#
# **할 일**:
# - `cosine_similarity(embeddings)` 로 유사도 행렬을 계산하세요.
# - DataFrame 으로 출력하고, **과제 3의 TF-IDF 유사도 행렬과 비교**하세요.

# %%
# 임베딩 벡터 간 코사인 유사도 계산
embedding_similarity = cosine_similarity(embeddings)

df_embed_sim = pd.DataFrame(
    embedding_similarity.round(3),
    index=[f"뉴스{i+1}" for i in range(len(finance_news))],
    columns=[f"뉴스{i+1}" for i in range(len(finance_news))],
)
print("[임베딩 기반 코사인 유사도 행렬]")
print(df_embed_sim)

# %% [markdown]
# **관찰 포인트**
# - TF-IDF 행렬과 달리 **0인 값이 거의 없습니다**.
# - 단어가 겹치지 않아도 의미가 비슷하면(예: 주가·주식·시장 관련 뉴스) 유사도가 올라갑니다.
# - 임베딩은 **단어 표면이 아니라 의미**를 기준으로 문장을 비교하기 때문입니다.

# %% [markdown]
# ---
# ## 과제 6. 핵심 비교 — 단어 매칭 vs 의미 유사도
#
# 이번 실습의 가장 중요한 부분입니다.
# **의미는 같지만 단어가 다른 문장 쌍**을 TF-IDF 와 임베딩으로 각각 비교합니다.
#
# **할 일**:
# - 아래 각 문장 쌍에 대해 TF-IDF 유사도와 임베딩 유사도를 계산해 나란히 출력하세요.
# - 두 수치가 **언제 크게 벌어지는지** 관찰하세요.

# %%
# 의미는 비슷하지만 표현(단어)이 다른 문장 쌍들
test_cases = [
    {'text1': '삼성전자 주가가 상승했습니다',
     'text2': '삼성전자 주식 가격이 올랐습니다'},          # 의미 같음, 단어 거의 다름
    {'text1': '주가 상승 실적 호조 전망 낙관',
     'text2': '주식 가격 증가 실적 좋음 전망 긍정적'},      # 의미 유사, 표현 다름
    {'text1': '반도체 시장 성장',
     'text2': '반도체 수요 증가'},                          # 일부 단어만 겹침
]

for case in test_cases:
    text1, text2 = case['text1'], case['text2']

    # TF-IDF 방식 — 두 문장만으로 벡터화 후 유사도
    tfidf_test = TfidfVectorizer()
    tfidf_vectors = tfidf_test.fit_transform([text1, text2])
    tfidf_sim = cosine_similarity(tfidf_vectors[0:1], tfidf_vectors[1:2])[0][0]

    # 임베딩 방식 — KURE-v1 로 인코딩 후 유사도
    emb_vectors = model.encode([text1, text2])
    emb_sim = cosine_similarity([emb_vectors[0]], [emb_vectors[1]])[0][0]

    print("-" * 70)
    print(f"  문장1: {text1}")
    print(f"  문장2: {text2}")
    print(f"  TF-IDF 유사도: {tfidf_sim:.4f}")
    print(f"  임베딩 유사도: {emb_sim:.4f}")
print("-" * 70)

# %% [markdown]
# **관찰 포인트 — 두 방식이 갈리는 지점**
# - 첫 번째 쌍처럼 **공유 단어가 거의 없으면** TF-IDF 유사도는 0에 가깝게 떨어집니다.
# - 같은 쌍이라도 임베딩 유사도는 **0.8 이상** 으로 높게 나옵니다.
# - TF-IDF 는 '주가' 와 '주식' 을 완전히 다른 단어로 보지만,
#   임베딩은 둘이 **의미적으로 가깝다는 것을 학습으로 알고 있기** 때문입니다.

# %% [markdown]
# ---
# ## 종합 정리
#
# 같은 문장들을 3가지 방식으로 벡터화하고 유사도를 비교했습니다.
#
# | 방법 | 벡터화 함수 | 벡터 형태 | 유사도 판단 기준 |
# |------|------------|----------|-----------------|
# | BOW | `CountVectorizer().fit_transform()` | 희소 (정수 빈도) | 같은 단어가 겹치는가 |
# | TF-IDF | `TfidfVectorizer().fit_transform()` | 희소 (실수 가중치) | 같은 단어가 겹치는가 |
# | 임베딩 | `SentenceTransformer().encode()` | 밀집 (1024차원) | 의미가 비슷한가 |
#
# **언제 어떤 방법을 쓸까?**
#
# | 상황 | 추천 | 이유 |
# |------|------|------|
# | 빠른 키워드 추출, 해석 용이성 중요 | BOW / TF-IDF | 처리 빠르고 어떤 단어가 기여했는지 보임 |
# | 의미 기반 검색·추천 | 임베딩 | 단어가 달라도 의미로 매칭 |
# | 대량 문서를 빠르게 처리 | TF-IDF | 계산 비용 낮음 |
# | 정확한 의미 분석이 중요 | 임베딩 | 문맥·의미 이해 |
#
# **핵심 메시지**: TF-IDF 는 **단어 표면의 일치**, 임베딩은 **의미의 유사성**을 잰다.
# 단어가 겹치지 않아도 의미가 같은 문장을 찾아야 한다면 임베딩이 필요합니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `finance_news` 에 자신만의 뉴스 문장을 1~2개 추가하고, TF-IDF·임베딩 유사도 행렬이
#    어떻게 바뀌는지 관찰하세요.
# 2. BOW 벡터(`bow_features`)에도 `cosine_similarity` 를 적용해 TF-IDF 유사도와 비교하세요.
# 3. `test_cases` 에 **의미가 완전히 다른 문장 쌍**(예: '주가 상승' vs '날씨가 맑다')을 추가하고,
#    임베딩 유사도가 실제로 낮게 나오는지 확인하세요.
# 4. `TfidfVectorizer(ngram_range=(1, 2))` 처럼 옵션을 바꿔 벡터가 어떻게 달라지는지 실험하세요.

# %%
