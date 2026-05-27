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
# # 140. 수강생 실습 — BERT Fine-tuning (네이버 영화 리뷰 감성분석)
#
# ## 학습 목표
# 사전학습된 다국어 BERT(`bert-base-multilingual-cased`) 를 NSMC (네이버 영화 리뷰)
# 데이터로 fine-tuning 해서 **한국어 문장의 긍정/부정을 판별하는 감성분석 모델**을
# 직접 만들어 봅니다.
#
# ## 이 실습에서 다루는 것
# - HuggingFace `Trainer` 를 이용한 **Sequence Classification** fine-tuning 의 전체 흐름
# - 사전학습 토크나이저로 한국어 문장을 BERT 입력(`input_ids` / `attention_mask` /
#   `token_type_ids`)으로 변환하는 방법
# - PyTorch `Dataset` 을 직접 정의해 Trainer 에 연결하는 방법
# - fine-tuning 된 모델로 새 문장에 대해 추론하는 방법
#
# ## 145 (NER) 실습과의 핵심 차이
# | 항목 | **140. 감성분석 (Sequence Classification)** | 145. NER (Token Classification) |
# |------|---------------------------------------------|----------------------------------|
# | 모델 클래스 | `AutoModel/BertForSequenceClassification` | `AutoModelForTokenClassification` |
# | 예측 단위 | 문장 1개 → 라벨 1개 (`[CLS]` 토큰만 사용) | 토큰마다 라벨 1개 (**모든 토큰** 사용) |
# | 라벨 형태 | 0(부정) / 1(긍정) — 정수 1개 | BIO 태그 시퀀스 |
#
# ## 감성분석 워크플로우 4단계
# ```
#   ① 데이터 로드  →  ② 토크나이즈 + Dataset 변환  →  ③ Trainer fine-tuning  →  ④ 추론
# ```
#
# > **실행 환경**: Colab GPU 기준 약 20 분 소요됩니다. CPU 만 있으면 매우 오래 걸리므로
# > 학습 샘플 수를 더 줄이거나 Colab GPU 환경에서 실행하세요.

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.

# %%
# !pip install -q transformers tensorflow

# %% [markdown]
# ---
# ## 1. 라이브러리 import 및 환경 확인

# %%
# Hugging Face Transformers 에서 BERT 토크나이저 / 분류 모델 / Trainer 로드
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import tensorflow as tf
import torch
import pandas as pd

# GPU(CUDA) 우선, 없으면 CPU 로 fallback
device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch 버전 :", torch.__version__)
print("사용 디바이스 :", device)
if device == "cuda":
    print("GPU 모델     :", torch.cuda.get_device_name(0))

# %% [markdown]
# ---
# ## 과제 1. NSMC 데이터 로드 및 결측치 처리
#
# NSMC (Naver Sentiment Movie Corpus) 는 네이버 영화 리뷰에 긍정(1) / 부정(0)
# 라벨이 달린 한국어 감성분석 벤치마크입니다. 학습 15만 + 검증 5만 문장.
#
# **할 일**:
# - `tf.keras.utils.get_file` 로 학습/검증 데이터를 다운로드하세요 (캐시됨).
# - `pd.read_csv(..., delimiter='\t')` 로 데이터프레임을 만들고 형태를 확인하세요.
# - `dropna(inplace=True)` 로 결측치(NaN) 가 포함된 행을 제거하세요.
#
# **힌트**: 다운로드 URL 은 이 강의 저자(`ironmanciti/Infran_NLP`)의 GitHub 에 미리
# 올라가 있습니다 — 새 URL 을 만들지 말고 기존 URL 을 그대로 사용하세요.

# %%
# NSMC 학습/검증 데이터 다운로드 (캐시됨)
DATA_TRAIN_PATH = tf.keras.utils.get_file(
    "ratings_train.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt")
DATA_TEST_PATH = tf.keras.utils.get_file(
    "ratings_test.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt")

# %% [markdown]
# ### Train Set

# %%
# 학습 데이터 로드
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter='\t')

print(train_data.shape)
train_data.head()

# %%
# 결측값(NaN)이 포함된 행을 모두 제거
train_data.dropna(inplace=True)

# 현재 DataFrame 의 구조 요약 출력
train_data.info()

# %% [markdown]
# ### Test Set

# %%
# 검증 데이터 로드
test_data = pd.read_csv(DATA_TEST_PATH, delimiter='\t')

print(test_data.shape)
test_data.head()

# %%
# 결측값(NaN)이 포함된 행을 모두 제거
test_data.dropna(inplace=True)
test_data.info()

# %% [markdown]
# **관찰 포인트**
# - 컬럼은 `id` / `document` / `label` 3개입니다. 우리는 `document`(문장)와
#   `label`(0=부정, 1=긍정) 만 사용합니다.
# - `dropna` 전후로 행 수가 약간 줄어드는 것을 확인하세요 — 결측치를 그대로 두면
#   tokenizer 호출 시 에러가 납니다.

# %% [markdown]
# ---
# ## 과제 2. 데이터 샘플링 및 X / y 분리
#
# 전체 15만 문장으로 학습하면 Colab 에서도 시간이 오래 걸립니다.
# 실습용으로 **학습 2만 / 검증 5천** 으로 축소합니다.
#
# **할 일**:
# - `sample(n=..., random_state=1)` 로 학습/검증 데이터를 무작위 추출하세요.
# - `document` 컬럼을 `X_train`, `X_test` 리스트로, `label` 컬럼을 `y_train`,
#   `y_test` 리스트로 분리하세요.
# - 학습 데이터의 긍정/부정 분포(`value_counts()`)를 출력해 클래스 균형을 확인하세요.
#
# **힌트**: `random_state=1` 을 지정해야 재실행해도 같은 샘플이 추출됩니다.

# %%
# 훈련 데이터에서 무작위로 20,000개 샘플 추출 (재현성을 위해 random_state 고정)
df_train = train_data.sample(n=20_000, random_state=1)

# 테스트 데이터에서 무작위로 5,000개 샘플 추출
df_test = test_data.sample(n=5_000, random_state=1)

# 추출된 데이터프레임의 행과 열 크기 출력
print(df_train.shape)
print(df_test.shape)

# %%
# 훈련 데이터의 'label' 열에 있는 각 클래스(레이블)별 개수를 집계
df_train['label'].value_counts()

# %%
# 훈련 데이터에서 입력 문장(document)과 레이블(label)을 리스트로 추출
X_train = df_train['document'].values.tolist()      # 입력 텍스트 (리스트 형태)
y_train = df_train['label'].values.tolist()         # 정답 레이블 (리스트 형태)

# 테스트 데이터에서도 동일하게 입력과 레이블을 리스트로 추출
X_test = df_test['document'].values.tolist()    # 입력 텍스트 (리스트 형태)
y_test = df_test['label'].values.tolist()       # 정답 레이블 (리스트 형태)

# %% [markdown]
# **관찰 포인트**
# - 긍정/부정 라벨의 개수가 거의 1:1 로 균형 잡혀 있어, 별도의 클래스 가중치
#   조정 없이 단순 fine-tuning 만 해도 됩니다.
# - 시간이 충분하다면 `n=20_000` 을 늘려 학습량을 키우고 성능 변화를 비교해 보세요.

# %% [markdown]
# ---
# ## 과제 3. 다국어 BERT 토크나이저 로드 및 토큰화
#
# `bert-base-multilingual-cased` 는 100개 이상 언어를 지원하는 다국어 BERT 입니다.
# 한국어 전용 모델은 아니지만, 사전학습 어휘에 한국어 sub-word 가 포함되어 있어
# 한국어 감성분석에서도 충분한 성능을 냅니다.
#
# **할 일**:
# - `BertTokenizer.from_pretrained('bert-base-multilingual-cased')` 로 토크나이저를 로드하세요.
# - `X_train`, `X_test` 를 `truncation=True, padding=True` 옵션으로 인코딩하세요.
# - 첫 샘플의 `input_ids`, `attention_mask`, `token_type_ids` 를 출력해 구조를 확인하세요.
#
# **힌트**:
# - `input_ids` : 토큰 인덱스, 모델의 실제 입력
# - `token_type_ids` : 두 문장(예: QA) 을 구분할 때 사용 (한 문장이면 모두 0)
# - `attention_mask` : 1=실제 토큰, 0=패딩 — 모델이 어디까지 봐야 하는지 알려줌
#
# ```
# [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
# ex) [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ```

# %%
# 사전학습된 BERT 토크나이저 불러오기
# 'bert-base-multilingual-cased' 는 100개 이상의 언어를 지원하는 다국어 BERT 모델로,
# 대소문자 구분(cased)을 유지함
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# %%
# 훈련 데이터(X_train)를 BERT 입력 형식에 맞게 토크나이즈
# - truncation=True: 최대 길이를 초과하는 문장은 자동으로 자름
# - padding=True   : 짧은 문장은 최대 길이에 맞춰 0으로 패딩
train_encodings = tokenizer(X_train, truncation=True, padding=True)

# 테스트 데이터(X_test) 도 동일한 방식으로 토크나이즈
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# %%
# 토크나이징된 훈련 데이터의 자료형 확인
# 일반적으로 'input_ids', 'attention_mask', 'token_type_ids' 키가 포함됨
print(type(train_encodings))

# %%
# 첫 샘플의 세 가지 시퀀스를 출력해 구조를 직접 확인
print(train_encodings['input_ids'][0])
print(train_encodings['attention_mask'][0])
print(train_encodings['token_type_ids'][0])

# %% [markdown]
# **관찰 포인트**
# - `input_ids` 의 첫 토큰은 항상 `[CLS]`(101), 끝은 `[SEP]`(102) 입니다.
# - `attention_mask` 가 `1` 인 구간이 실제 문장이고, 뒤쪽 `0` 들은 패딩입니다.
# - `token_type_ids` 는 단일 문장 분류라 모두 `0` 입니다 (QA 처럼 두 문장을 입력할 때만 `1` 이 등장).

# %% [markdown]
# ---
# ## 과제 4. PyTorch Dataset 클래스 정의
#
# Trainer 가 학습에 사용하려면 입력을 `torch.utils.data.Dataset` 형태로 감싸야 합니다.
# `__len__` 과 `__getitem__` 두 메서드만 구현하면 됩니다.
#
# **할 일**:
# - `IMDbDataset` 클래스를 정의해 토크나이즈된 인코딩과 라벨을 묶으세요.
# - `__getitem__` 은 dict 를 반환해야 하며 키는 `input_ids`, `attention_mask`,
#   `token_type_ids`, `labels` 입니다 — 키 이름이 정확해야 Trainer 가 알아챕니다.
# - 학습/검증용 Dataset 객체를 각각 생성하세요.
#
# **힌트**: `torch.tensor(...)` 로 감싸는 이유는 Trainer 가 배치 단위로 텐서를 모아
# GPU 에 올리기 때문입니다.

# %%
# PyTorch Dataset 클래스를 상속하여 NSMC 감성분석용 커스텀 데이터셋 정의
class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        # 토크나이즈된 입력 (input_ids, attention_mask 등) 저장
        self.encodings = encodings
        # 정답 레이블 (선택사항)
        self.labels = labels

    def __getitem__(self, idx):
        # 주어진 인덱스(idx)에 해당하는 데이터 추출
        # encodings 딕셔너리에서 각 항목별로 같은 인덱스를 추출하고 텐서로 변환
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # 레이블이 있는 경우 함께 반환 (Trainer 는 'labels' 키를 자동으로 찾아 loss 계산)
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        # 데이터셋의 전체 샘플 수 반환
        return len(self.encodings["input_ids"])


# 훈련용 PyTorch Dataset 객체 생성
train_dataset = IMDbDataset(train_encodings, y_train)

# 테스트용 PyTorch Dataset 객체 생성
test_dataset = IMDbDataset(test_encodings, y_test)

# %% [markdown]
# **관찰 포인트**
# - `labels` 키 하나만 있으면 Trainer 가 모델 출력 logits 와 비교해 자동으로
#   CrossEntropyLoss 를 계산합니다. 직접 loss 함수를 작성할 필요가 없습니다.
# - 같은 클래스에 추론용 데이터셋(라벨 없음) 을 만들고 싶다면 `labels=None` 으로
#   호출하면 됩니다.

# %% [markdown]
# ---
# ## 과제 5. TrainingArguments + 모델 로드 + Trainer 학습
#
# **할 일**:
# - `TrainingArguments` 로 하이퍼파라미터(epochs, batch size, weight decay 등)를 설정하세요.
# - `BertForSequenceClassification.from_pretrained(..., num_labels=2)` 로 모델을 로드하세요.
# - `Trainer` 객체를 만들고 `trainer.train()` 을 호출하세요.
#
# **힌트**:
# - `warmup_ratio=0.1` — 전체 스텝의 10% 동안 학습률을 천천히 올린 뒤(웜업) 본 학습.
#   매우 작은 학습률로 시작해 학습 초기 불안정을 줄입니다.
# - `weight_decay=0.01` — L2 정규화 계수.
# - `num_labels=2` — 긍정/부정 두 클래스. 명시하지 않으면 기본값 `2` 가 적용되지만,
#   다중분류 확장을 염두에 두고 명시적으로 적는 습관을 들이세요.

# %%
training_args = TrainingArguments(
    output_dir='./results',               # 모델 출력 결과(가중치 등) 저장 디렉토리
    num_train_epochs=2,                   # 학습 전체 epoch 수
    per_device_train_batch_size=16,       # 학습 시 디바이스(GPU/CPU) 당 배치 크기
    per_device_eval_batch_size=16,        # 평가 시 디바이스당 배치 크기
    warmup_ratio=0.1,                     # 전체 스텝의 10% 를 워밍업에 사용
    weight_decay=0.01,                    # 가중치 감쇠(L2 정규화) 계수
    logging_dir='./logs',                 # 로그 저장 디렉토리
    logging_steps=100,                    # 몇 스텝마다 로그를 출력할지 설정
    report_to="none",                     # wandb 등 모든 로깅 통합 비활성화
    run_name='naver_movie_sentiment'      # 명시적 run_name 설정
)

# %% [markdown]
# ### 모델 학습 (약 20분 소요)

# %%
import time

# 사전학습된 다국어 BERT 모델 로드 (문장 분류용 head 가 새로 붙음)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=2  # 긍정/부정 2개 클래스 명시적 지정
)

# Hugging Face 의 Trainer 객체 생성
trainer = Trainer(
    model=model,                  # 학습할 모델
    args=training_args,           # 학습 설정 (TrainingArguments 객체)
    train_dataset=train_dataset,  # 훈련 데이터셋
    eval_dataset=test_dataset     # 평가 데이터셋
)

# 학습 시작 시간 기록
s = time.time()

# 모델 학습 수행
trainer.train()

# %%
print("경과 시간 : {:.2f}분".format((time.time() - s) / 60))

# %% [markdown]
# **관찰 포인트**
# - 학습 로그에서 `loss` 가 epoch 가 진행됨에 따라 감소하는지 확인하세요.
# - `BertForSequenceClassification` 의 분류 head 만 새로 초기화되고, BERT 본체는
#   사전학습 가중치 그대로 시작해 점진적으로 fine-tuning 됩니다.

# %% [markdown]
# ---
# ## 과제 6. 모델 평가
#
# **할 일**:
# - `trainer.evaluate(test_dataset)` 으로 종합 성능을 확인하세요.
# - `trainer.predict(test_dataset)` 으로 logit 을 얻고, softmax 후 argmax 로
#   예측 라벨을 만드세요.
# - `accuracy_score`, `confusion_matrix` 로 정확도와 혼동 행렬을 출력하세요.
#
# **힌트**: fine-tuned 모델은 **logit (정규화 전 점수)** 을 반환합니다.
# 확률이 필요하면 `F.softmax(logits, dim=-1)` 를 적용해야 합니다.

# %%
# 테스트 데이터셋을 사용하여 모델 성능 평가
# 반환값에는 손실(loss), 정확도 등의 평가 지표가 포함됨
trainer.evaluate(test_dataset)

# %%
# 테스트 데이터셋에 대해 예측 수행
# 출력은 예측 결과(predictions), 실제 정답(label_ids), 평가 지표(metrics)를 포함한 객체
prediction = trainer.predict(test_dataset)
prediction

# %%
# 현재 Trainer 에 포함된 모델의 분류기(classifier) 층 확인
# 이 층은 BERT 출력(hidden state)을 받아 최종 분류 결과를 계산하는 레이어
trainer.model.classifier

# %%
# 모델 예측 결과에서 로짓(logits) 값을 텐서로 변환
# prediction[0] 은 trainer.predict() 의 결과 중 'predictions' (로짓 값)
y_logit = torch.tensor(prediction[0])

# 처음 10개 샘플의 로짓 출력
# 각 샘플마다 클래스 수만큼의 점수(예: 2-class 분류면 [logit0, logit1]) 가 있음
y_logit[:10]

# %%
# 소프트맥스 함수를 사용해 각 샘플의 클래스별 확률을 계산
# dim=-1     : 마지막 차원(클래스 차원) 기준으로 소프트맥스 적용
# argmax(axis=1) : 확률이 가장 높은 클래스의 인덱스를 예측값으로 선택
# numpy()    : PyTorch 텐서를 넘파이 배열로 변환
y_pred = F.softmax(y_logit, dim=-1).argmax(axis=1).numpy()

# 예측된 레이블 중 앞 30개를 리스트로 출력
print(list(y_pred[:30]))

# 실제 정답 레이블(y_test) 중 앞 30개를 출력
print(y_test[:30])

# %%
from sklearn.metrics import confusion_matrix, accuracy_score

# 예측값과 실제 정답 사이의 정확도(accuracy)를 계산
print(accuracy_score(y_test, y_pred))

# 혼동 행렬(confusion matrix) 계산
# 실제 레이블과 예측 레이블을 비교하여 각 클래스별 예측 결과를 표로 요약
cm = confusion_matrix(y_test, y_pred)
cm

# %% [markdown]
# **관찰 포인트**
# - 2 epoch 학습만으로도 정확도가 보통 **0.85 이상** 나옵니다.
# - 혼동 행렬의 대각선이 두꺼울수록 좋습니다. 비대각 항목은 오분류를 의미합니다.
# - 데이터 샘플을 늘리거나 epoch 를 더 돌리면 성능이 더 올라갑니다.

# %% [markdown]
# ---
# ## 과제 7. Fine-tuned 모델로 새 문장 추론
#
# 학습이 끝났으면 새 한국어 문장을 직접 넣어 긍정/부정 판단을 받아 봅니다.
#
# **할 일**:
# - 새 문장을 토크나이즈해 `return_tensors="pt"` 로 텐서를 만드세요.
# - `model.eval()` 로 평가 모드 전환 후, 입력 텐서를 모델과 같은 디바이스로 옮기세요.
# - `torch.no_grad()` 블록에서 추론하고, softmax → argmax 로 최종 라벨을 얻으세요.
#
# **힌트**: 학습 때와 달리 입력이 1개 문장이어도 `tokenizer([x], ...)` 처럼
# **리스트로 감싸야** 배치 차원이 만들어집니다.

# %%
# 예측할 문장
x = "돈주고 보기에는 아까운 영화 ㅠㅠ..."
# x = "내 인생 최고 명작"

# 1. 입력 토크나이즈
inputs = tokenizer([x], truncation=True, padding=True, return_tensors="pt")

# 2. 입력을 모델과 같은 디바이스로 이동
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {k: v.to(device) for k, v in inputs.items()}

# 3. 추론
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 4. 소프트맥스 → 확률 → argmax
probs = F.softmax(logits, dim=-1)
pred = torch.argmax(probs, dim=1).item()

# 5. 결과 출력
print("긍정" if pred == 1 else "부정")

# %% [markdown]
# **관찰 포인트**
# - "돈주고 보기에는 아까운 영화" → 부정, "내 인생 최고 명작" → 긍정 으로 잡혀야 합니다.
# - 학습 데이터에 없던 신조어·이모티콘이 섞여도, BERT 가 sub-word 토큰화 덕분에
#   어느 정도 일반화해 분류해 줍니다.

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 도구 | 감성분석 고유 포인트 |
# |------|-----------|----------------------|
# | ① 데이터 로드 | `pd.read_csv` + `dropna` | NSMC 는 0(부정)/1(긍정) 이진 라벨 |
# | ② 토크나이즈 | `BertTokenizer(truncation, padding)` | `[CLS] ... [SEP]` + `attention_mask` |
# | ③ Dataset 정의 | `torch.utils.data.Dataset` 상속 | `labels` 키만 있으면 Trainer 가 loss 자동 계산 |
# | ④ fine-tuning | `BertForSequenceClassification` + `Trainer` | `[CLS]` 토큰 출력만 분류에 사용 |
# | ⑤ 추론 | `model(**inputs)` → `softmax` → `argmax` | logits 를 직접 확률로 변환 |
#
# **핵심 메시지**:
# - Sequence Classification 은 문장 전체를 대표하는 `[CLS]` 토큰의 임베딩 하나로
#   라벨을 예측합니다 (145 NER 의 token-level 예측과 대비).
# - HuggingFace `Trainer` 덕분에 학습 루프를 직접 짜지 않고도 다양한 옵션(warmup,
#   weight decay, logging) 을 설정으로 제어할 수 있습니다.
# - 다국어 사전학습 모델만으로도 한국어 task 에서 충분히 좋은 baseline 이 나옵니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `df_train` 의 `n=20_000` 을 `50_000` 또는 전체로 늘려 다시 학습하고,
#    정확도가 얼마나 올라가는지 비교하세요.
# 2. `num_train_epochs` 를 `3 ~ 5` 로 늘려 보고, 어느 시점부터 검증 loss 가 다시
#    오르는지(과적합 시점) 확인하세요.
# 3. 본인이 작성한 한국어 영화 리뷰 5문장을 과제 7 에 추가해 모델 판단을 확인하세요.
#    특히 반어법·이모티콘이 섞인 문장에서 모델이 약한 지점을 찾아보세요.
# 4. `model.save_pretrained("./naver-sentiment-bert")` 와
#    `tokenizer.save_pretrained("./naver-sentiment-bert")` 로 저장한 뒤,
#    새 셀에서 다시 `from_pretrained` 로 로드해 동일한 추론 결과가 나오는지 확인하세요.

# %%
