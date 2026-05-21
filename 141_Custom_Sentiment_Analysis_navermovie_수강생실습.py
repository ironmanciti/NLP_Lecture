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
# # 141. 수강생 실습 - BERT Fine-Tuning 으로 한국어 감성분석 모델 만들기
#
# ## 학습 목표
# 사전학습된 **다국어 BERT** 를 네이버 영화 리뷰(NSMC) 데이터로 **fine-tuning** 하여,
# 리뷰 문장의 **긍정/부정**을 판별하는 감성분석 모델을 직접 만듭니다.
#
# HuggingFace `Trainer` 를 사용한 fine-tuning 의 전체 흐름을 경험하는 것이 목표입니다.
#
# ```
# 원본 데이터 → 토큰화 → PyTorch Dataset → Trainer 학습 → 평가 → 새 문장 예측
# ```
#
# **실행 환경**: Colab GPU 권장 (과제 4의 학습에 약 20분 소요됩니다).

# %%
# 실습에 필요한 패키지 설치 (Colab 기준 — 로컬에 이미 설치돼 있으면 생략 가능)
# !pip install -q transformers

# %%
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import tensorflow as tf
import torch
import pandas as pd

# %% [markdown]
# ---
# ## 과제 1. 데이터 준비하기
#
# NSMC(네이버 영화 리뷰)는 리뷰 문장과 긍정(1)/부정(0) 라벨로 이루어진 데이터셋입니다.
#
# **할 일**:
# - `tf.keras.utils.get_file` 로 학습용/테스트용 데이터를 내려받으세요.
# - `pandas` 로 읽은 뒤 결측치(`dropna`)를 제거하세요.
# - 학습 시간 단축을 위해 훈련 20,000개 / 테스트 5,000개만 무작위 추출하세요.
# - `document`(문장)와 `label`(정답)을 각각 리스트로 분리하세요.
#
# **힌트**: 전체 15만 건은 Colab GPU 로도 오래 걸리므로 일부만 표본 추출합니다.
# `random_state` 를 고정하면 매번 같은 표본이 뽑혀 결과를 재현할 수 있습니다.

# %%
# NSMC 학습/테스트 데이터 다운로드 (탭으로 구분된 txt)
DATA_TRAIN_PATH = tf.keras.utils.get_file(
    "ratings_train.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt")
DATA_TEST_PATH = tf.keras.utils.get_file(
    "ratings_test.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt")

# 데이터 로드 후 결측치 제거
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter='\t').dropna()
test_data = pd.read_csv(DATA_TEST_PATH, delimiter='\t').dropna()

# 학습 시간 단축을 위해 일부만 무작위 추출 (random_state 고정 → 재현 가능)
df_train = train_data.sample(n=20_000, random_state=1)
df_test = test_data.sample(n=5_000, random_state=1)
print("훈련 데이터:", df_train.shape, " 테스트 데이터:", df_test.shape)

# 라벨 분포 확인 (0=부정, 1=긍정)
print("\n[훈련 데이터 라벨 분포]")
print(df_train['label'].value_counts())

# 입력 문장(X)과 정답 라벨(y)을 리스트로 분리
X_train = df_train['document'].values.tolist()
y_train = df_train['label'].values.tolist()
X_test = df_test['document'].values.tolist()
y_test = df_test['label'].values.tolist()

# %% [markdown]
# **관찰 포인트**
# - `label` 은 **0(부정) / 1(긍정)** 두 가지뿐인 **이진 분류** 문제입니다.
# - 두 클래스의 개수가 비슷하면 데이터가 균형 잡혀 있어 정확도를 신뢰하기 좋습니다.

# %% [markdown]
# ---
# ## 과제 2. BERT 토크나이저로 문장 토큰화하기
#
# BERT 는 문자열을 그대로 받지 못하므로, 사전학습된 토크나이저로
# **정수 ID 시퀀스**로 변환해야 합니다.
#
# **할 일**:
# - `bert-base-multilingual-cased` 토크나이저를 불러오세요 (한국어 포함 100여 개 언어 지원).
# - `X_train`, `X_test` 를 `truncation=True, padding=True` 옵션으로 토큰화하세요.
# - 첫 번째 샘플의 `input_ids` 와 `attention_mask` 를 출력해 보세요.
#
# **힌트**: BERT 입력은 `[CLS] 문장 [SEP]` 구조입니다.
# `attention_mask` 의 `1` 은 실제 토큰, `0` 은 패딩 자리를 뜻합니다.

# %%
# 사전학습된 다국어 BERT 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# 문장들을 토큰화 (truncation: 너무 길면 자름, padding: 짧으면 0으로 채움)
train_encodings = tokenizer(X_train, truncation=True, padding=True)
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# 첫 번째 샘플의 토큰화 결과 확인
print("input_ids     :", train_encodings['input_ids'][0])
print("attention_mask:", train_encodings['attention_mask'][0])

# %% [markdown]
# **관찰 포인트**
# - `input_ids` 는 토큰을 정수로 바꾼 시퀀스이며, 맨 앞 `101`(`[CLS]`)·중간 `102`(`[SEP]`)가 특수 토큰입니다.
# - 모든 문장이 같은 길이가 되도록 짧은 문장 뒤쪽이 **0으로 패딩**됩니다.
# - `attention_mask` 가 0인 위치는 패딩이므로 모델이 무시합니다.

# %% [markdown]
# ---
# ## 과제 3. PyTorch Dataset 으로 변환하기
#
# HuggingFace `Trainer` 는 토큰화 결과(딕셔너리)가 아니라 **PyTorch Dataset 객체**를 입력으로 받습니다.
#
# **할 일**:
# - `torch.utils.data.Dataset` 을 상속하는 클래스를 만들고 `__getitem__`, `__len__` 을 구현하세요.
# - 토큰화 결과와 라벨을 묶어 `train_dataset`, `test_dataset` 을 생성하세요.
#
# **힌트**: `__getitem__` 은 인덱스 하나에 해당하는 `{입력, 라벨}` 딕셔너리를 텐서로 반환해야 합니다.

# %%
# PyTorch Dataset 클래스 정의 — 토큰화 결과와 라벨을 하나로 묶는다
class NaverMovieDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings   # 토큰화 결과 (input_ids, attention_mask 등)
        self.labels = labels         # 정답 라벨 (없을 수도 있음)

    def __getitem__(self, idx):
        # idx 번째 샘플의 각 항목을 텐서로 변환해 딕셔너리로 반환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

# 훈련용/테스트용 Dataset 객체 생성
train_dataset = NaverMovieDataset(train_encodings, y_train)
test_dataset = NaverMovieDataset(test_encodings, y_test)
print("train_dataset 샘플 수:", len(train_dataset))
print("한 샘플의 키:", list(train_dataset[0].keys()))

# %% [markdown]
# **관찰 포인트**
# - `Dataset` 은 "인덱스를 주면 그 위치의 데이터 한 개를 돌려주는" 규격입니다.
# - `Trainer` 는 내부에서 이 객체를 배치 단위로 꺼내 모델에 넣습니다.

# %% [markdown]
# ---
# ## 과제 4. 학습 설정과 BERT Fine-Tuning
#
# 이제 사전학습 BERT 에 **분류기(classifier) head** 를 붙여 우리 데이터로 학습시킵니다.
#
# **할 일**:
# - `TrainingArguments` 로 학습 설정(epoch 수, 배치 크기 등)을 정의하세요.
# - `BertForSequenceClassification` 으로 모델을 로드하세요 (`num_labels=2` — 긍정/부정).
# - `Trainer` 를 만들고 `trainer.train()` 으로 학습을 시작하세요.
#
# **힌트**: Colab GPU 기준 약 20분 소요됩니다.
# `warmup_ratio` 는 학습 초반에 학습률을 천천히 올려 안정적으로 출발하게 합니다.

# %%
# 학습 설정 정의
training_args = TrainingArguments(
    output_dir='./results',            # 결과(가중치) 저장 경로
    num_train_epochs=2,                # 전체 학습 epoch 수
    per_device_train_batch_size=16,    # 학습 배치 크기
    per_device_eval_batch_size=16,     # 평가 배치 크기
    warmup_ratio=0.1,                  # 전체 스텝의 10%를 워밍업에 사용
    weight_decay=0.01,                 # 가중치 감쇠 (L2 정규화)
    logging_dir='./logs',
    logging_steps=100,
    report_to="none",                  # wandb 등 외부 로깅 비활성화
    run_name='naver_movie_sentiment',
)

# %%
import time

# 사전학습 다국어 BERT + 분류기 head 로드 (num_labels=2 → 긍정/부정)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased', num_labels=2)

# Trainer 생성 — 모델 / 설정 / 데이터셋을 묶는다
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 학습 시작
s = time.time()
trainer.train()
print("경과 시간 : {:.2f}분".format((time.time() - s) / 60))

# %% [markdown]
# **관찰 포인트**
# - 모델은 이미 언어를 아는 **사전학습 BERT 본체** + 새로 초기화된 **분류기 head** 로 구성됩니다.
# - Fine-tuning 은 전체를 처음부터 학습하는 것이 아니라,
#   이미 똑똑한 모델을 **우리 과제(감성분석)에 맞게 미세 조정**하는 것입니다.
# - 그래서 비교적 적은 데이터·짧은 시간으로도 높은 성능이 나옵니다.

# %% [markdown]
# ---
# ## 과제 5. 모델 평가하기
#
# 학습된 모델이 테스트 데이터에서 얼마나 잘 맞히는지 확인합니다.
#
# **할 일**:
# - `trainer.predict(test_dataset)` 로 예측 결과(로짓)를 얻으세요.
# - 로짓에 softmax → argmax 를 적용해 예측 라벨을 구하세요.
# - `accuracy_score` 와 `confusion_matrix` 로 성능을 평가하세요.
#
# **힌트**: 모델의 출력은 확률이 아니라 **로짓(logit)** 입니다.
# `softmax` 로 확률로 바꾼 뒤 `argmax` 로 가장 점수가 높은 클래스를 고릅니다.

# %%
from sklearn.metrics import accuracy_score, confusion_matrix

# 테스트 데이터에 대해 예측 수행
prediction = trainer.predict(test_dataset)

# 로짓 → softmax(확률) → argmax(예측 라벨)
y_logit = torch.tensor(prediction[0])
y_pred = F.softmax(y_logit, dim=-1).argmax(axis=1).numpy()

# 정확도와 혼동 행렬 출력
print("정확도(accuracy):", accuracy_score(y_test, y_pred))
print("\n[혼동 행렬]  행=실제, 열=예측")
print(confusion_matrix(y_test, y_pred))

# %% [markdown]
# **관찰 포인트**
# - 혼동 행렬의 대각선(좌상·우하)이 정답, 나머지가 오답 개수입니다.
# - 사전학습 모델을 단 2 epoch fine-tuning 했을 뿐인데도 정확도가 꽤 높게 나옵니다.
#   → **전이학습(transfer learning)** 의 힘입니다.

# %% [markdown]
# ---
# ## 과제 6. 새로운 문장으로 감성 예측하기
#
# 학습된 모델로 임의의 새 리뷰 문장이 긍정인지 부정인지 예측해 봅니다.
#
# **할 일**:
# - 예측할 문장을 토크나이즈하세요 (`return_tensors="pt"`).
# - `model.eval()` 과 `torch.no_grad()` 안에서 추론하세요.
# - 로짓에 softmax → argmax 를 적용해 "긍정"/"부정" 을 출력하세요.
#
# **힌트**: 추론 시에는 `model.eval()` 로 평가 모드로 바꾸고,
# `torch.no_grad()` 로 기울기 계산을 꺼서 메모리·속도를 아낍니다.

# %%
# 예측할 문장 (자유롭게 바꿔 보세요)
x = "돈주고 보기에는 아까운 영화 ㅠㅠ..."
# x = "내 인생 최고 명작"

# 입력 토크나이즈 → 모델과 같은 디바이스로 이동
inputs = tokenizer([x], truncation=True, padding=True, return_tensors="pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 평가 모드로 추론
model.eval()
with torch.no_grad():
    logits = model(**inputs).logits

# softmax → argmax → 결과 출력
pred = torch.argmax(F.softmax(logits, dim=-1), dim=1).item()
print(f"입력 문장: {x}")
print("예측 결과:", "긍정" if pred == 1 else "부정")

# %% [markdown]
# ---
# ## 종합 정리
#
# 사전학습 BERT 를 fine-tuning 해 한국어 감성분석 모델을 완성했습니다.
#
# | 단계 | 핵심 도구 | 하는 일 |
# |------|----------|---------|
# | 1. 데이터 준비 | `pandas` | 리뷰·라벨 로드 및 표본 추출 |
# | 2. 토큰화 | `BertTokenizer` | 문장 → 정수 ID 시퀀스 |
# | 3. Dataset 변환 | `torch.utils.data.Dataset` | Trainer 가 읽을 형식으로 포장 |
# | 4. Fine-Tuning | `BertForSequenceClassification` + `Trainer` | 사전학습 모델을 감성분석에 맞게 조정 |
# | 5. 평가 | `accuracy_score`, `confusion_matrix` | 테스트 정확도 측정 |
# | 6. 예측 | `model(**inputs)` | 새 문장의 긍정/부정 판별 |
#
# **핵심 메시지**: 밑바닥부터 학습하지 않고, **이미 언어를 아는 사전학습 모델을 재활용**하는 것이
# fine-tuning(전이학습)입니다. 적은 데이터로도 빠르게 좋은 성능을 얻을 수 있습니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. 과제 6의 `x` 에 자신만의 리뷰 문장을 여러 개 넣어 예측 결과를 확인하세요.
# 2. `num_train_epochs` 를 3으로 늘리거나 표본 수를 키워 정확도가 어떻게 변하는지 비교하세요.
# 3. 사전학습 모델을 `klue/bert-base`(한국어 특화)로 바꿔 학습하고 성능을 비교하세요.
# 4. 예측이 틀린 테스트 문장 몇 개를 직접 찾아 출력하고, 왜 틀렸을지 생각해 보세요.
# 5. 전체 20만 건 데이터로 fine-tuning 하면 성능이 얼마나 올라갈지 예상해 보세요.

# %%
