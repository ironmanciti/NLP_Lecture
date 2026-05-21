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
# # 151. 수강생 실습 - BERT Fine-Tuning 으로 한국어 개체명 인식(NER) 모델 만들기
#
# ## 학습 목표
# 사전학습 모델 `klue/bert-base` 를 **개체명 인식(NER, Named Entity Recognition)** task로
# fine-tuning 하여, 한국어 문장에서 인물·장소·기관 등을 자동으로 찾아내는 모델을 만듭니다.
#
# ```
# 데이터셋 로드 → 토큰화 + 라벨 정렬 → 모델 로드 → Trainer 학습 → 평가 → 추론
# ```
#
# ## 감성분석(141) 실습과의 핵심 차이
# | 항목 | 141. 감성분석 (문장 분류) | 151. NER (토큰 분류) |
# |------|--------------------------|----------------------|
# | 모델 클래스 | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` |
# | 예측 단위 | 문장 1개 → 라벨 1개 | **토큰마다** 라벨 1개 |
# | 라벨 형태 | 문장 단위 정수 (긍정/부정) | 토큰 단위 BIO 태그 시퀀스 |
#
# ## BIO 태그란?
# NER 라벨은 토큰마다 태그를 하나씩 붙이는 **BIO 태깅** 방식을 씁니다.
#
# | 태그 | 의미 |
# |------|------|
# | **B-** (Begin) | 개체명이 **시작**되는 토큰 (예: `B-LC` = 장소 시작) |
# | **I-** (Inside) | 개체명이 **이어지는** 토큰 (예: `I-LC` = 장소가 계속됨) |
# | **O** (Outside) | 개체명이 **아닌** 일반 토큰 |
#
# 예) `"이순신 장군은 서울 용산구에 산다"` → `이순신`=B-PS, `서울`=B-LC, `용산구에`=I-LC, 나머지=O
#
# KLUE-NER 은 6개 개체 유형(PS LC OG DT TI QT)을 다루므로
# `6 × 2(B-, I-) + O = 13개` 라벨이 됩니다.
#
# **실행 환경**: Colab GPU 권장 (과제 4의 학습에 수십 분 소요될 수 있습니다).

# %%
# 실습에 필요한 패키지 설치 (Colab 기준 — 로컬에 이미 설치돼 있으면 생략 가능)
# !pip install -q transformers datasets evaluate seqeval accelerate

# %%
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
import evaluate

# GPU(CUDA) 우선, 없으면 CPU 로 fallback
device = "cuda" if torch.cuda.is_available() else "cpu"
print("사용 디바이스 :", device)

# %% [markdown]
# ---
# ## 과제 1. KLUE-NER 데이터셋 로드 및 라벨 구조 파악하기
#
# KLUE-NER 은 한국어 NER 벤치마크 데이터셋입니다.
# - `tokens` : **글자(character) 단위**로 분리된 리스트
# - `ner_tags` : 각 글자에 대응하는 BIO 태그 (정수로 인코딩됨)
#
# **할 일**:
# - `load_dataset("klue", "ner")` 로 데이터셋을 불러오세요.
# - 첫 번째 학습 샘플의 `tokens` 와 `ner_tags` 를 출력해 구조를 확인하세요.
# - 라벨 목록(`label_list`)과 `id2label`, `label2id` 변환표를 만드세요.
#
# **힌트**: `ner_tags` 의 정수값은 데이터셋 feature 의 순서로 인코딩되어 있으므로,
# 라벨 목록은 **반드시 feature 에서 가져와야** 합니다 (하드코딩 금지).

# %%
# KLUE-NER 데이터셋 로드 (train / validation split)
raw_datasets = load_dataset("klue", "ner")
print(raw_datasets)

# 라벨 목록을 데이터셋 feature 에서 가져온다
label_list = raw_datasets["train"].features["ner_tags"].feature.names
num_labels = len(label_list)
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}

# 첫 번째 학습 샘플의 구조 확인
sample = raw_datasets["train"][0]
print("\n문장        :", sample["sentence"])
print("토큰(앞 15) :", sample["tokens"][:15])
print("태그(앞 15) :", [label_list[t] for t in sample["ner_tags"][:15]])

print(f"\n라벨 개수 : {num_labels}")
print(f"라벨 목록 : {label_list}")

# %% [markdown]
# **관찰 포인트**
# - `tokens` 는 단어가 아니라 **글자 하나하나**로 쪼개져 있고, 글자마다 BIO 태그가 붙습니다.
# - 라벨은 총 **13개** (`O` + 6개 유형 × `B-`/`I-`).
# - 개체가 아닌 글자는 모두 `O` 이므로, 실제로는 `O` 가 대부분을 차지합니다.

# %% [markdown]
# ---
# ## 과제 2. 토크나이저 준비와 sub-word 라벨 정렬(alignment)
#
# **이 실습에서 가장 중요한 부분입니다.**
# KLUE-NER 라벨은 **글자 단위**로 붙어 있지만, BERT 토크나이저는 단어를 **sub-word** 로
# 다시 쪼갭니다. 따라서 "원래 글자 라벨"을 "sub-word 토큰"에 다시 맞춰 줘야(align) 합니다.
#
# **할 일**:
# - `klue/bert-base` 토크나이저를 로드하세요.
# - `tokenize_and_align_labels` 함수를 이해하고, 전체 데이터셋에 `map` 으로 적용하세요.
#
# **정렬 규칙**:
# - 한 단어의 **첫 sub-word** → 원래 라벨 부여
# - 같은 단어의 **이어지는 sub-word** → `-100`
# - **특수 토큰**(`[CLS]`, `[SEP]`, padding) → `-100`
#
# **힌트**: `-100` 은 PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값이라,
# 해당 위치는 loss 계산에서 자동으로 무시됩니다.

# %%
# klue/bert-base 사전학습 토크나이저 로드
model_checkpoint = "klue/bert-base"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def tokenize_and_align_labels(examples):
    """글자 단위 라벨을 sub-word 토큰에 맞춰 정렬한다."""
    # tokens 가 이미 분리된 리스트이므로 is_split_into_words=True 로 지정
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True)

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        # word_ids(): 각 sub-word 가 원래 몇 번째 단어에서 왔는지 알려준다
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)                  # 특수 토큰 → 무시
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])      # 단어의 첫 sub-word → 원래 라벨
            else:
                label_ids.append(-100)                  # 이어지는 sub-word → 무시
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


# %%
# 변환 전/후를 직접 비교해 alignment 동작을 눈으로 확인한다
example = raw_datasets["train"][0]
demo = tokenize_and_align_labels(
    {"tokens": [example["tokens"]], "ner_tags": [example["ner_tags"]]})
demo_tokens = tokenizer.convert_ids_to_tokens(demo["input_ids"][0])

print(f"{'sub-word 토큰':<16}{'정렬된 라벨'}")
print("-" * 32)
for tok, lab in zip(demo_tokens, demo["labels"][0]):
    lab_str = id2label[lab] if lab != -100 else "-100 (무시)"
    print(f"{tok:<16}{lab_str}")

# %%
# map() 으로 전체 데이터셋에 alignment 를 일괄 적용한다
# remove_columns: 원본 컬럼(tokens, ner_tags, sentence)은 학습에 불필요하므로 제거
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
print(tokenized_datasets)

# %% [markdown]
# **관찰 포인트**
# - 맨 앞 `[CLS]`, 맨 뒤 `[SEP]` 는 `-100` 으로 표시됩니다 → loss 계산에서 제외.
# - 한 글자가 여러 sub-word 로 쪼개졌다면, 첫 조각만 라벨을 받고 나머지는 `-100`.
# - 이 정렬을 빼먹으면 라벨과 토큰의 위치가 어긋나 모델이 엉뚱한 것을 학습합니다.

# %% [markdown]
# ---
# ## 과제 3. Token Classification 모델 로드하기
#
# **할 일**:
# - `AutoModelForTokenClassification` 으로 모델을 로드하세요.
# - `num_labels`, `id2label`, `label2id` 를 함께 전달하세요.
#
# **힌트**: 141 감성분석은 `[CLS]` 토큰 한 자리에만 classifier 를 적용했지만,
# NER 은 classifier head 가 **모든 토큰 위치**에 적용됩니다.

# %%
# token classification 용 head 가 붙은 모델 로드 (head 는 새로 초기화됨)
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# classifier head 출력: (hidden_size=768) → (num_labels=13) 선형 변환
print("[ classifier head ]")
print(model.classifier)

# %% [markdown]
# **관찰 포인트**
# - classifier 는 `768 → 13` 선형 변환입니다. 각 토큰의 768차원 표현을 13개 라벨 점수로 바꿉니다.
# - 사전학습된 BERT 본체는 그대로 재사용하고, classifier head 만 새로 학습합니다.

# %% [markdown]
# ---
# ## 과제 4. Trainer 구성 및 Fine-Tuning
#
# **할 일**:
# - `DataCollatorForTokenClassification` 으로 배치 내 동적 패딩을 준비하세요.
# - `seqeval` 기반 `compute_metrics` 함수로 평가 지표(precision/recall/F1)를 정의하세요.
# - `TrainingArguments` 와 `Trainer` 를 만들고 `trainer.train()` 으로 학습하세요.
#
# **힌트**: NER 성능은 단순 정확도가 아니라 **entity 단위 F1** 으로 평가합니다.
# `seqeval` 은 BIO 시퀀스를 개체 단위로 묶어 점수를 계산해 줍니다.

# %%
# DataCollator: 배치 안에서 input_ids 와 labels 를 같은 길이로 동적 패딩
# (labels 의 패딩 자리는 자동으로 -100 으로 채워져 loss 에서 제외됨)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# seqeval 기반 평가 지표 함수
seqeval = evaluate.load("seqeval")


def compute_metrics(eval_preds):
    """모델 예측(logits)과 정답(labels)으로 NER 성능 지표를 계산한다."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # -100 으로 마스킹된 위치는 제외하고 정수 id 를 문자열 BIO 태그로 변환
    true_labels = [
        [id2label[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [id2label[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# %%
# 학습/평가 데이터셋 준비
# 시간이 너무 오래 걸리면 아래 select() 주석을 풀어 일부만 사용하세요.
# train_dataset = tokenized_datasets["train"].select(range(2000))
# eval_dataset  = tokenized_datasets["validation"].select(range(500))
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["validation"]
print(f"학습 샘플 수 : {len(train_dataset)}, 평가 샘플 수 : {len(eval_dataset)}")

# 학습 하이퍼파라미터 설정
training_args = TrainingArguments(
    output_dir="./klue-ner-results",
    eval_strategy="epoch",                # 매 epoch 마다 검증셋 평가
    save_strategy="epoch",                # 매 epoch 마다 체크포인트 저장
    learning_rate=5e-5,
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,          # 학습 후 가장 좋은 체크포인트 복원
    metric_for_best_model="f1",           # "가장 좋은" 기준은 F1
    report_to="none",
)

# Trainer 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
# 모델 fine-tuning 실행
import time

start_time = time.time()
trainer.train()
print(f"학습 소요 시간 : {(time.time() - start_time) / 60:.1f}분")

# %% [markdown]
# **관찰 포인트**
# - epoch 가 진행될수록 검증 F1 이 올라가는지 로그를 확인하세요.
# - `load_best_model_at_end=True` 덕분에 학습 종료 시 F1 이 가장 높았던 체크포인트가 복원됩니다.

# %% [markdown]
# ---
# ## 과제 5. 모델 평가 — entity 유형별 성능 확인하기
#
# **할 일**:
# - `trainer.evaluate()` 로 검증셋 종합 지표를 출력하세요.
# - `seqeval.metrics.classification_report` 로 개체 유형별(PS/LC/OG/DT/TI/QT) 성능을 확인하세요.

# %%
from seqeval.metrics import classification_report

# 검증셋 종합 성능
eval_results = trainer.evaluate()
print("[ 종합 성능 ]")
for key in ["eval_precision", "eval_recall", "eval_f1", "eval_accuracy"]:
    print(f"  {key:18s}: {eval_results[key]:.4f}")

# 개체 유형별 상세 성능
predictions, labels, _ = trainer.predict(eval_dataset)
predictions = np.argmax(predictions, axis=-1)

true_labels = [
    [id2label[l] for l in label if l != -100]
    for label in labels
]
true_predictions = [
    [id2label[p] for p, l in zip(prediction, label) if l != -100]
    for prediction, label in zip(predictions, labels)
]

print("\n[ 개체 유형별 성능 ]")
print(classification_report(true_labels, true_predictions))

# %% [markdown]
# **관찰 포인트**
# - 유형마다 F1 이 다릅니다 — 학습 데이터에 많이 등장한 유형일수록 보통 점수가 높습니다.
# - accuracy 는 높지만 F1 은 그보다 낮을 수 있습니다.
#   `O` 토큰이 대부분이라 정확도는 쉽게 올라가지만, **실제 개체를 잡는 능력은 F1** 이 보여줍니다.

# %% [markdown]
# ---
# ## 과제 6. Fine-Tuned 모델로 추론하기
#
# **할 일**:
# - `pipeline("ner", ...)` 로 추론 파이프라인을 만드세요.
# - `aggregation_strategy="simple"` 을 주어 같은 개체의 연속 토큰을 하나로 묶으세요.
# - 새로운 한국어 문장 몇 개에 개체명 인식을 실행하세요.

# %%
# fine-tuned 모델로 NER 파이프라인 구성
ner_pipeline = pipeline(
    "ner",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",        # 연속 토큰을 하나의 entity 로 병합
    device=0 if device == "cuda" else -1,
)

# 개체 유형 코드를 한국어 명칭으로 변환하는 표
entity_kor = {"PS": "인물", "LC": "장소", "OG": "기관",
              "DT": "날짜", "TI": "시간", "QT": "수량"}

# 새 문장에 개체명 인식 실행 (자유롭게 바꿔 보세요)
examples = [
    "삼성전자는 2024년 3월 서울 강남구에 새 사옥을 열었다.",
    "이순신 장군은 1592년 한산도에서 일본 수군을 격파했다.",
]

for text in examples:
    print("=" * 55)
    print(f"문장: {text}")
    entities = ner_pipeline(text)
    if entities:
        for ent in entities:
            kind = entity_kor.get(ent["entity_group"], ent["entity_group"])
            print(f"   [{kind}] {ent['word']}  (신뢰도 {ent['score']:.2f})")
    else:
        print("   (인식된 개체 없음)")
print("=" * 55)

# %% [markdown]
# **관찰 포인트**
# - `aggregation_strategy="simple"` 이 `B-`/`I-` 로 이어진 토큰을 하나의 단어로 합쳐 줍니다.
# - 학습 데이터(KLUE-NER)와 비슷한 문체일수록 인식이 정확합니다.

# %% [markdown]
# ---
# ## 종합 정리
#
# 사전학습 BERT 를 fine-tuning 해 한국어 개체명 인식(NER) 모델을 완성했습니다.
#
# | 단계 | 핵심 도구 | NER 고유 포인트 |
# |------|-----------|------------------|
# | 1. 데이터 로드 | `load_dataset("klue", "ner")` | 라벨이 글자 단위 BIO 태그 |
# | 2. 토큰화 + 정렬 | `tokenize_and_align_labels` + `word_ids()` | sub-word alignment, `-100` 마스킹 |
# | 3. 모델 로드 | `AutoModelForTokenClassification` | classifier 가 **모든 토큰**에 적용 |
# | 4. Fine-Tuning | `DataCollatorForTokenClassification` + `Trainer` | seqeval 로 entity 단위 F1 평가 |
# | 5. 평가 | `classification_report` | 개체 유형별 성능 분석 |
# | 6. 추론 | `pipeline("ner", aggregation_strategy="simple")` | 연속 토큰을 entity 로 병합 |
#
# **핵심 메시지**: NER 은 **토큰마다 라벨을 매기는** 토큰 분류 문제입니다.
# 가장 까다로운 부분은 글자 단위 라벨을 sub-word 토큰에 맞추는 **alignment** 이며,
# 이 단계를 정확히 처리하는 것이 NER fine-tuning 의 성패를 가릅니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. 과제 6의 `examples` 에 자신만의 문장을 넣어 개체명 인식 결과를 확인하세요.
# 2. 과제 2의 정렬 규칙을 바꿔, 이어지는 sub-word 에도 `-100` 대신 같은 라벨을 부여하면
#    성능이 어떻게 달라지는지 실험하세요.
# 3. `num_train_epochs` 를 늘리거나 줄여 F1 변화를 비교하세요.
# 4. `model.save_pretrained("./klue-ner-bert-base")` 와 `tokenizer.save_pretrained(...)` 로
#    모델을 저장하고, `pipeline("ner", model="./klue-ner-bert-base")` 로 다시 불러와 보세요.

# %%
