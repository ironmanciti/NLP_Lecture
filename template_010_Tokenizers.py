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
# # 010. 토크나이저 실습

# %%
# KoNLPy(한국어 형태소 분석기 패키지) 설치

# %%

# %% [markdown]
# # 1. Keras 기본 Tokenizer - rule-based
# - 공백 또는 구둣점으로 분리  
# - 영어 단어별로 띄어쓰기가 철저히 지켜지는 언어

# %%
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
# 주어진 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
# 구축된 단어 인덱스 사전 가져오기
# 단어 인덱스 사전 출력

# %% [markdown]
# Keras의 rule base tokenizer로 한글을 tokenize

# %%
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
# 주어진 한글 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
# 구축된 단어 인덱스 사전 가져오기
# 단어 인덱스 사전 출력

# %% [markdown]
# # 2. 단어 사전 기반 한국어 tokenizer 사용

# %%
# Okt 형태소 분석기 객체 생성
# 형태소 분석 결과를 저장할 리스트 초기화
# 주어진 한글 문장 리스트의 각 문장에 대해 반복
    # 문장을 형태소 분석하여 결과를 리스트에 추가
    # 형태소 분석 결과 출력

# %% [markdown]
# 사전 기반 tokenize 후 Keras tokenizer 로 vocabulary 생성

# %%
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
# 형태소 분석된 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
# 구축된 단어 인덱스 사전 가져오기
# 단어 인덱스 사전 출력

# %% [markdown]
# 두 vocabulary 의 차이 비교

# %%

# %% [markdown]
# ### 단, Okt 사전에 미등록된 단어의 경우 정확한 tokenizing 이 안된다.

# %%
# 주어진 문장을 형태소 분석하여 품사 태깅 수행

# %% [markdown]
# 예를 들어 `너무너무너무`와 `나카무라세이코`는 하나의 단어이지만, okt 사전에 등록되어 있지 않아 여러 개의 복합단어로 나뉘어집니다. 이러한 문제를 해결하기 위하여 형태소 분석기와 품사 판별기들은 사용자 사전 추가 기능을 제공합니다. 사용자 사전을 추가하여 모델의 vocabulary 를 풍부하게 만드는 것은 사용자의 몫입니다.
#
# 1. okt 공식 문서를 참고해서 사용사 사전을 추가.
# 2. okt를 패키징하고, konlpy에서 사용할 수 있도록 konlpy/java 경로에 jar 파일을 복사.
# 3. 기존에 참고하고 있던 okt.jar 대신 새로운 okt.jar를 사용하도록 설정.
# 4. konlpy 소스 경로를 import 해서 형태소 분석.

# %% [markdown]
# # 3. Google SentencePiece Tokenizer
#
# - NAVER Movie rating data 를 이용한 sentencepiece tokenizer training

# %%

# %% [markdown]
# - pandas.read_csv에서 quoting = 3으로 설정해주면 인용구(따옴표)를 무시

# %%

# %%

# %%

# %% [markdown]
# ## 학습을 위해 text 를 따로 저장

# %%
# 'nsmc.txt' 파일을 쓰기 모드로 열기 (UTF-8 인코딩 사용)
    # 훈련 데이터의 'document' 열에 있는 각 문장에 대해 반복
            # 문장을 파일에 쓰고 새로운 줄 추가
            # 쓰기 오류 발생 시 오류 메시지와 해당 문장 출력

# %%
#write 가 잘 되었는지 확인

# %%
# 명령어 템플릿 정의
# 템플릿에 변수 값을 포맷하여 명령어 문자열 생성

# %% [markdown]
# ### sentencepiece tokenizer training

# %%
# SentencePieceTrainer를 사용하여 SentencePiece 모델 학습

# %%
# SentencePieceProcessor 객체 생성
# 학습된 SentencePiece 모델 로드

# %%
# 훈련 데이터의 'document' 열에 있는 첫 세 개의 문장에 대해 반복
    # 원본 문장 출력
    # 문장을 SentencePiece 모델을 사용하여 토큰화하여 출력
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환하여 출력

# %%
# 한글 문장 리스트(sentences_K)에 있는 각 문장에 대해 반복
    # 문장을 SentencePiece 모델을 사용하여 토큰화
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환
    # 원본 문장 출력
    # 토큰화된 결과 출력
    # 인덱스 시퀀스 출력
    # 각 문장 사이에 줄 바꿈 추가

# %% [markdown]
# # OpenAI의 tiktoken을 사용한 tokenizer 개념 이해
#
# - tiktoken은 OpenAI에서 개발한 빠르고 효율적인 BPE(Byte Pair Encoding) 기반 토크나이저입니다.  
# - GPT 모델들이 사용하는 것과 동일한 토크나이징 방식을 제공합니다.
#
# ```
#     "cl100k_base": "GPT-4, GPT-3.5-turbo, text-embedding-ada-002에서 사용",
#     "p50k_base": "GPT-3, Codex에서 사용",
#     "r50k_base": "GPT-3, GPT-2에서 사용"
# ```
#
# - tiktoken은 BPE(Byte Pair Encoding) 방식을 사용합니다. 가장 자주 등장하는 문자 쌍을 하나의 토큰으로 병합하는 방식입니다.

# %%
# cl100k_base 인코더 사용
# 영어 문장 리스트 순회
    # 텍스트를 토큰 ID 리스트로 인코딩
    # 토큰 ID에 해당하는 실제 토큰 출력
    # 토큰 ID를 다시 텍스트로 복원
    # 토큰 개수 출력

    # %%
    # 텍스트를 토큰 ID 리스트로 인코딩
    # 토큰 ID에 해당하는 실제 토큰 출력
    # 토큰 ID를 다시 텍스트로 복원
    # 토큰 개수 출력

# %%
