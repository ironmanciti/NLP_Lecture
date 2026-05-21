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
# !pip install -q KoNLPy

# %%
sentences_E = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'I was born in Korea and graduaged University in USA.',
]

sentences_K = [
    "코로나가 심하다",
    "코비드-19가 심하다",
    '아버지가방에들어가신다',
    '아버지가 방에 들어가신다',
    '너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다'
]

# %% [markdown]
# # 1. Keras 기본 Tokenizer - rule-based
# - 공백 또는 구둣점으로 분리  
# - 영어 단어별로 띄어쓰기가 철저히 지켜지는 언어

# %%
from tensorflow.keras.preprocessing.text import Tokenizer

# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 주어진 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(sentences_E)

# 구축된 단어 인덱스 사전 가져오기
word_index = tokenizer.word_index

# 단어 인덱스 사전 출력
print(word_index)

# %% [markdown]
# Keras의 rule base tokenizer로 한글을 tokenize

# %%
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 주어진 한글 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(sentences_K)

# 구축된 단어 인덱스 사전 가져오기
vocabulary_keras_korean = tokenizer.word_index

# 단어 인덱스 사전 출력
print(vocabulary_keras_korean)

# %% [markdown]
# # 2. 단어 사전 기반 한국어 tokenizer 사용

# %%
from konlpy.tag import Okt

# Okt 형태소 분석기 객체 생성
okt = Okt()

# 형태소 분석 결과를 저장할 리스트 초기화
temp_X = []

# 주어진 한글 문장 리스트의 각 문장에 대해 반복
for sent in sentences_K:
    # 문장을 형태소 분석하여 결과를 리스트에 추가
    temp_X.append(okt.morphs(sent))
    # 형태소 분석 결과 출력
    print(okt.morphs(sent))

# %% [markdown]
# 사전 기반 tokenize 후 Keras tokenizer 로 vocabulary 생성

# %%
# 빈도수 상위 100개의 단어로 구성된 Tokenizer 객체 생성 (OOV(Out-Of-Vocabulary) 토큰 설정)
tokenizer = Tokenizer(num_words=100, oov_token='<OOV>')

# 형태소 분석된 문장 리스트에 대해 토크나이저 학습 수행 (단어 인덱스 구축)
tokenizer.fit_on_texts(temp_X)

# 구축된 단어 인덱스 사전 가져오기
vocabulary_okt_keras = tokenizer.word_index

# 단어 인덱스 사전 출력
print(vocabulary_okt_keras)

# %% [markdown]
# 두 vocabulary 의 차이 비교

# %%
print(vocabulary_keras_korean)
print(vocabulary_okt_keras)

# %% [markdown]
# ### 단, Okt 사전에 미등록된 단어의 경우 정확한 tokenizing 이 안된다.

# %%
# 주어진 문장을 형태소 분석하여 품사 태깅 수행
okt.pos('너무너무너무는 나카무라세이코가 불러 크게 히트한 노래입니다')

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
import tensorflow as tf
import pandas as pd
import sentencepiece as spm

DATA_TRAIN_PATH = tf.keras.utils.get_file("ratings_train.txt",
        "https://github.com/ironmanciti/infran_NLP/raw/main/data/naver_movie/ratings_train.txt")

# %% [markdown]
# - pandas.read_csv에서 quoting = 3으로 설정해주면 인용구(따옴표)를 무시

# %%
train_data = pd.read_csv(DATA_TRAIN_PATH, sep='\t', quoting=3)

print(train_data.shape)
train_data.head()

# %%
train_data.isnull().sum()

# %%
train_data.dropna(inplace=True)

train_data.shape

# %% [markdown]
# ## 학습을 위해 text 를 따로 저장

# %%
# 'nsmc.txt' 파일을 쓰기 모드로 열기 (UTF-8 인코딩 사용)
with open('./nsmc.txt', 'w', encoding='utf-8') as f:
    # 훈련 데이터의 'document' 열에 있는 각 문장에 대해 반복
    for line in train_data.document.values:
        try:
            # 문장을 파일에 쓰고 새로운 줄 추가
            f.write(line + '\n')
        except:
            # 쓰기 오류 발생 시 오류 메시지와 해당 문장 출력
            print("write error ---> ", line)

# %%
#write 가 잘 되었는지 확인
with open('./nsmc.txt', 'r', encoding='utf-8') as f:
    nsmc_txt = f.read().split('\n')

print(len(nsmc_txt))
print(nsmc_txt[0])

# %%
input_file = 'nsmc.txt'   # 입력 파일 경로 설정
vocab_size = 30000    # 어휘 사전의 최대 크기 설정
prefix = 'nsmc'             # 모델 파일의 접두사 설정

# 명령어 템플릿 정의
templates = '--input={} --model_prefix={} --vocab_size={}'

# 템플릿에 변수 값을 포맷하여 명령어 문자열 생성
cmd = templates.format(input_file, prefix, vocab_size)
cmd

# %% [markdown]
# ### sentencepiece tokenizer training

# %%
# SentencePieceTrainer를 사용하여 SentencePiece 모델 학습
spm.SentencePieceTrainer.Train(cmd)

# %%
# SentencePieceProcessor 객체 생성
sp = spm.SentencePieceProcessor()

# 학습된 SentencePiece 모델 로드
sp.Load('{}.model'.format(prefix))

# %%
# 훈련 데이터의 'document' 열에 있는 첫 세 개의 문장에 대해 반복
for t in train_data.document.values[:3]:
    # 원본 문장 출력
    print(t)
    # 문장을 SentencePiece 모델을 사용하여 토큰화하여 출력
    print(sp.encode_as_pieces(t))
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환하여 출력
    print(sp.encode_as_ids(t), '\n')

# %%
# 한글 문장 리스트(sentences_K)에 있는 각 문장에 대해 반복
for line in sentences_K:
    # 문장을 SentencePiece 모델을 사용하여 토큰화
    pieces = sp.encode_as_pieces(line)
    # 문장을 SentencePiece 모델을 사용하여 인덱스 시퀀스로 변환
    ids = sp.encode_as_ids(line)
    # 원본 문장 출력
    print(line)
    # 토큰화된 결과 출력
    print(pieces)
    # 인덱스 시퀀스 출력
    print(ids)
    # 각 문장 사이에 줄 바꿈 추가
    print()

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
import tiktoken
import pandas as pd
import numpy as np

# cl100k_base 인코더 사용
encoding = tiktoken.get_encoding("cl100k_base")

# 영어 문장 리스트 순회
for i, sentence in enumerate(sentences_E):
    print(f"\n문장 {i+1}: {sentence}")

    # 텍스트를 토큰 ID 리스트로 인코딩
    tokens = encoding.encode(sentence)
    print(f"토큰 ID: {tokens}")

    # 토큰 ID에 해당하는 실제 토큰 출력
    token_texts = [
        encoding.decode_single_token_bytes(token).decode("utf-8", errors="replace")
        for token in tokens
    ]
    print(f"토큰 문자열: {token_texts}")

    # 토큰 ID를 다시 텍스트로 복원
    decoded_text = encoding.decode(tokens)
    print(f"디코딩 결과: {decoded_text}")

    # 토큰 개수 출력
    token_count = len(tokens)
    print(f"토큰 개수: {token_count}")

# %%
for i, sentence in enumerate(sentences_K):
    print(f"\n문장 {i+1}: {sentence}")

    # 텍스트를 토큰 ID 리스트로 인코딩
    tokens = encoding.encode(sentence)
    print(f"토큰 ID: {tokens}")

    # 토큰 ID에 해당하는 실제 토큰 출력
    token_texts = [
        encoding.decode_single_token_bytes(token).decode("utf-8", errors="replace")
        for token in tokens
    ]
    print(f"토큰 문자열: {token_texts}")

    # 토큰 ID를 다시 텍스트로 복원
    decoded_text = encoding.decode(tokens)
    print(f"디코딩 결과: {decoded_text}")

    # 토큰 개수 출력
    token_count = len(tokens)
    print(f"토큰 개수: {token_count}")

# %%
