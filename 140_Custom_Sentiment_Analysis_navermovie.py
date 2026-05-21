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
# # 320. Custom Dataset을 이용한 Hugging Face BERT model Fine Tuning
#
# - NAVER Movie review dataset을 이용하여 transformers BERT model을 fine tuning  → 감성분석 모델 작성
#
# - Pytorch 와 Trainer를 이용한 Fine Tuning (Pytorch version이 Tensorflow 보다 안정적)
#
# - Colab gpu 사용

# %%
# Hugging Face Transformers에서 BERT 토크나이저 로드
from transformers import BertTokenizer
# Hugging Face Transformers에서 BERT 기반 문장 분류 모델 로드
from transformers import BertForSequenceClassification, Trainer, TrainingArguments
import torch.nn.functional as F
import tensorflow as tf
import torch
import pandas as pd

# %%
DATA_TRAIN_PATH = tf.keras.utils.get_file("ratings_train.txt",
                     "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt")
DATA_TEST_PATH = tf.keras.utils.get_file("ratings_test.txt",
                    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt")

# %% [markdown]
# ### Train Set

# %%
#  학습 데이터 로드
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter='\t')

print(train_data.shape)
train_data.head()

# %%
# 결측값(NaN)이 포함된 행을 모두 제거
train_data.dropna(inplace=True)

# 현재 DataFrame의 구조 요약 출력
train_data.info()

# %% [markdown]
# ### Test Set

# %%
#  검증 데이터 로드
test_data = pd.read_csv(DATA_TEST_PATH, delimiter='\t')

print(test_data.shape)
test_data.head()

# %%
# 결측값(NaN)이 포함된 행을 모두 제거
test_data.dropna(inplace=True)
test_data.info()

# %% [markdown]
# - 훈련 시간 단축을 위해 data size 축소

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
y_train = df_train['label'].values.tolist()               # 정답 레이블 (리스트 형태)

# 테스트 데이터에서도 동일하게 입력과 레이블을 리스트로 추출
X_test = df_test['document'].values.tolist()    # 입력 텍스트 (리스트 형태)
y_test = df_test['label'].values.tolist()             # 정답 레이블 (리스트 형태)

# %% [markdown]
# ## pre-trained bert model 호출
# ### tokenizer 호출
# - 토큰화 처리를 합니다. bert 다국어 version 용의 pre-trained tokenizer 를 불러옵니다.

# %%
# 사전학습된 BERT 토크나이저 불러오기
# 'bert-base-multilingual-cased'는 100개 이상의 언어를 지원하는 다국어 BERT 모델로,
# 대소문자 구분(cased)을 유지함
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# %% [markdown]
# pre-trained tokenizer 를 이용하여 train set 과 test set 을 token 화 합니다.
#
# - Input IDs : 토큰 인덱스, 모델에서 입력으로 사용할 시퀀스를 구축하는 토큰의 숫자 표현
# - Token Type IDs : 한 쌍의 문장 또는 질문 답변에 대한 분류 시 사용  
# - attention mask : `1`은 주목해야 하는 값을 나타내고 `0`은 패딩된 값을 나타냅니다.  
# ```
# [CLS] SEQUENCE_A [SEP] SEQUENCE_B [SEP]
# ex) [CLS] HuggingFace is based in NYC [SEP] Where is HuggingFace based? [SEP]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# ```

# %%
# 훈련 데이터(X_train)를 BERT 입력 형식에 맞게 토크나이즈
# - truncation=True: 최대 길이를 초과하는 문장은 자동으로 자름
# - padding=True: 짧은 문장은 최대 길이에 맞춰 0으로 패딩
train_encodings = tokenizer(X_train, truncation=True, padding=True)

# 테스트 데이터(X_test)도 동일한 방식으로 토크나이즈
test_encodings = tokenizer(X_test, truncation=True, padding=True)

# %%
# 토크나이징된 훈련 데이터의 키 목록 확인
# 일반적으로 'input_ids', 'attention_mask', (선택적으로 'token_type_ids')가 포함됨
print(type(train_encodings))

# %%
print(train_encodings['input_ids'][0])
print(train_encodings['attention_mask'][0])
print(train_encodings['token_type_ids'][0])


# %% [markdown]
# ### Convert encodings to Tensors
#
# - 레이블과 인코딩을 Dataset 개체로 변환합니다. Pytorch를 이용합니다.  
#
# - PyTorch에서 이것은 `torch.utils.data.Dataset` 객체를 하고 `__len__` 및 `__getitem__`을 구현하여 수행됩니다.
#

# %%
# PyTorch Dataset 클래스를 상속하여 IMDb 감성 분석용 커스텀 데이터셋 정의
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
        # 레이블이 있는 경우 함께 반환
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
# 이제 데이터 세트가 준비되었으므로 🤗 `Trainer` 또는 기본 PyTorch/TensorFlow를 사용하여 모델을 미세 조정할 수 있습니다. [training](https://huggingface.co/transformers/training.html)을 참조하세요.
#
# - Training warmup steps :  
#
#     - 이는 일반적으로 설정된 수의 훈련 단계(워밍업 단계)에 대해 매우 낮은 학습률을 사용한다는 것을 의미합니다. 워밍업 단계 후에 "일반" 학습률 또는 학습률 스케줄러를 사용합니다. 또한 워밍업 단계 수에 따라 학습률을 점진적으로 높일 수 있습니다.
#
# - weight_decay : 가중치 감쇠. L2 regularization

# %%
training_args = TrainingArguments(
    output_dir='./results',               # 모델 출력 결과(가중치 등)를 저장할 디렉토리
    num_train_epochs=2,                   # 학습 전체 epoch 수
    per_device_train_batch_size=16,       # 학습 시 디바이스(GPU/CPU)당 배치 크기
    per_device_eval_batch_size=16,        # 평가 시 디바이스당 배치 크기
    warmup_ratio=0.1,                     # 전체 스텝의 10%를 워밍업에 사용 (batch 변경에 자동 대응)
    weight_decay=0.01,                    # 가중치 감쇠(정규화) 계수
    logging_dir='./logs',                 # 로그 저장 디렉토리
    logging_steps=100,                    # 몇 스텝마다 로그를 출력할지 설정
    report_to="none",                     # wandb 등 모든 로깅 통합 비활성화
    run_name='naver_movie_sentiment'      # 명시적 run_name 설정
)

# %% [markdown]
# ### model Train
#
# 약 20 분 소요

# %%
import time

# 사전학습된 다국어 BERT 모델 로드 (문장 분류용으로 head가 붙어 있음)
model = BertForSequenceClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=2  # 긍정/부정 2개 클래스 명시적 지정
)

# Hugging Face의 Trainer 객체 생성
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
print("경과 시간 : {:.2f}분".format((time.time() - s)/60))

# %%
# 테스트 데이터셋을 사용하여 모델 성능 평가
# 반환값에는 손실(loss), 정확도(accuracy) 등의 평가 지표가 포함됨
trainer.evaluate(test_dataset)

# %%
# 테스트 데이터셋에 대해 예측 수행
# 출력은 예측 결과(predictions), 실제 정답(label_ids), 평가 지표(metrics)를 포함한 객체
prediction = trainer.predict(test_dataset)
prediction

# %% [markdown]
# fine-tuned model 은 logit 을 return

# %%
# 현재 Trainer에 포함된 모델에서 분류기(classifier) 층 확인
# 이 층은 BERT 출력(hidden state)을 받아 최종 분류 결과를 계산하는 레이어
trainer.model.classifier

# %%
# 모델 예측 결과에서 로짓(logits) 값을 텐서로 변환
# prediction[0]은 trainer.predict()의 결과 중 'predictions' (로짓 값)
y_logit = torch.tensor(prediction[0])

# 처음 10개 샘플의 로짓 출력
# 각 샘플마다 클래스 수만큼의 점수(예: 2-class 분류면 [logit0, logit1])가 있음
y_logit[:10]

# %%
# 소프트맥스 함수를 사용해 각 샘플의 클래스별 확률을 계산
# dim=-1: 마지막 차원(클래스 차원) 기준으로 소프트맥스 적용
# argmax(axis=1): 확률이 가장 높은 클래스의 인덱스를 예측값으로 선택
# numpy(): PyTorch 텐서를 넘파이 배열로 변환
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

# %%
# 예측할 문장
x = "돈주고 보기에는 아까운 영화 ㅠㅠ..."
# x = "내 인생 최고 명작"

# 1. 입력 토크나이즈
inputs = tokenizer([x], truncation=True, padding=True, return_tensors="pt")

# 2. 입력을 GPU로 이동하고 예측
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
inputs = {k: v.to(device) for k, v in inputs.items()}  # 모든 입력을 같은 디바이스로 이동

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# 3. 소프트맥스 → 확률 → argmax
probs = F.softmax(logits, dim=-1)
pred = torch.argmax(probs, dim=1).item()

# 4. 결과 출력
print("긍정" if pred == 1 else "부정")

# %% [markdown]
# # Next Step
# 20 만개 전체 dataset으로 fine tuning
