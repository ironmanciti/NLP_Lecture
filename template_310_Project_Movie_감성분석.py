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
# # 딥러닝 기반 영화 리뷰 감성 분석 시스템
#
# ## 프로젝트 목표
# - 딥러닝 기법을 활용한 감성 분석 시스템 구축
# - 두 가지 방법 비교: Hugging Face Pipeline vs ClovaX
# - 네이버 영화평 데이터를 이용한 실전 감성 분석
#
# ## 학습 내용
# 1. 최소한의 데이터 전처리
# 2. 감성 분석 방법 1: Hugging Face Pipeline
# 3. 감성 분석 방법 2: ClovaX

# %% [markdown]
# ---
# ## 1. 데이터 준비 및 최소 전처리

# %%
# 환경 변수 로드

# %%
# 네이버 영화평 데이터 다운로드
# 데이터 로드
# 결측값 제거
# 데이터 샘플링
# 딥러닝 모델용 최소 전처리 (결측값 제거만)
# 빈 문자열 제거

# %% [markdown]
# ---
# ## 2. 감성 분석 방법 1: Hugging Face Pipeline

# %% [markdown]
# ### 2.1 파이프라인을 이용한 감성 분석

# %%
# 다국어 감성 분석 모델
# 샘플 리뷰 분석

# %% [markdown]
# ### 2.2 테스트 데이터 감성 분석

# %%
# 테스트 데이터 샘플 분석
    # 별점을 긍정/부정으로 변환 (4-5점: 긍정, 1-2점: 부정)


# %% [markdown]
# ---
# ## 3. 감성 분석 방법 2: ClovaX

# %% [markdown]
# ### 3.1 ClovaX를 사용한 감성 분석
#
# **지시사항**:
# - ClovaX 모델을 사용하여 감성 분석 수행
# - 프롬프트를 통해 긍정/부정 분류 요청
# - Hugging Face와 동일한 테스트 데이터로 평가
#
# %%

# %%
def analyze_sentiment_clovax(text):
    # 필요시 <|endofturn|>, <|stop|> 등에서 자르기
    # 생성된 텍스트에서 사용자 입력 부분 제거
    # "긍정" 또는 "부정" 키워드 확인
        # 기본값 (긍정으로 가정)
# 샘플 리뷰 분석

# %% [markdown]
#    ### 3.2 테스트 데이터 감성 분석
# %%
# 테스트 데이터 샘플 분석

# %%
