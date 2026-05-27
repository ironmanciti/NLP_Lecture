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
# # 145. 수강생 실습 — BERT NER Fine-tuning (KLUE-NER)
#
# ## 학습 목표
# 사전학습된 `klue/bert-base` 모델을 KLUE-NER 데이터셋으로 fine-tuning 해서
# 한국어 문장에서 **인물·장소·기관·날짜·시간·수량** 같은 개체명을 자동으로 인식하는
# 모델을 직접 만들어 봅니다.
#
# ## 140 (감성분석) 실습과의 핵심 차이
# | 항목 | 140. 감성분석 (Sequence Classification) | 145. NER (Token Classification) |
# |------|----------------------------------------|----------------------------------|
# | 모델 클래스 | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` |
# | 예측 단위 | 문장 1개 → 라벨 1개 (`[CLS]` 토큰만 사용) | 토큰마다 라벨 1개 (**모든 토큰** 사용) |
# | 라벨 형태 | 문장 단위 정수 (긍정/부정) | 토큰 단위 BIO 태그 시퀀스 |
#
# ## BIO 태그란?
# NER 라벨은 **BIO 태깅**으로 표현합니다. 토큰마다 태그를 하나씩 붙입니다.
#
# | 태그 | 의미 | 설명 |
# |------|------|------|
# | **B-** (Begin) | 개체명의 **시작** 토큰 | 예) `B-LC` = 장소 개체가 여기서 시작 |
# | **I-** (Inside) | 개체명의 **이어지는** 토큰 | 예) `I-LC` = 앞의 장소 개체가 계속됨 |
# | **O** (Outside) | 개체명이 **아닌** 토큰 | 조사·동사 등 일반 토큰 |
#
# 예시 — `"이순신 장군은 서울 용산구에 산다"`
#
# | 토큰 | 이순신 | 장군은 | 서울 | 용산구에 | 산다 |
# |------|--------|--------|------|----------|------|
# | BIO 태그 | **B-PS** | O | **B-LC** | **I-LC** | O |
#
# **13개 라벨 계산**: KLUE-NER 은 6개 개체 유형(PS LC OG DT TI QT)을 다루므로
# `6 × 2(B-, I-) + O 1개 = 13개` 라벨이 됩니다. → 뒤에서 `num_labels=13`.
#
# ## NER 워크플로우 4단계
# ```
#   ① 데이터셋 로드  →  ② 토크나이저 + 라벨 정렬  →  ③ Trainer fine-tuning  →  ④ 추론
# ```

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.

# %%

# %% [markdown]
# ---
# ## 1. 라이브러리 import 및 환경 확인

# %%
# GPU(CUDA) 우선, 없으면 CPU 로 fallback

# %% [markdown]
# ---
# ## 과제 1. KLUE-NER 데이터셋 로드 및 구조 확인
#
# KLUE-NER 은 한국어 NER 벤치마크입니다.
#
# **할 일**:
# - `load_dataset("klue", "ner")` 로 train / validation split 을 로드하세요.
# - 학습 데이터 첫 샘플의 `sentence`, `tokens`, `ner_tags` 를 출력해 구조를 확인하세요.
#
# **힌트**:
# - `tokens` 는 **글자(character) 단위** 리스트입니다.
# - `ner_tags` 는 각 글자에 대응하는 BIO 태그의 **정수 인코딩**입니다.
# - 라벨 이름(`B-PS`, `I-LC` 등)은 `raw_datasets["train"].features["ner_tags"].feature.names` 로 얻을 수 있습니다.

# %%
# KLUE-NER 데이터셋의 train / validation split 을 로드한다

# %%
# 학습 데이터 샘플 1~2개를 출력해 구조를 눈으로 확인한다
# 토큰(글자) · ner_tags(숫자) · 라벨명(알파벳)을 세로로 정렬해 한눈에 보이게 한다
# ner_tags 의 정수값 → 라벨명(B-LC, I-LC, O 등) 변환표
def disp_width(s):
def pad(s, width):
    # 토큰을 12개씩 끊어서 토큰 / 숫자 / 라벨 3줄을 칸 맞춰 출력
        # 공백 토큰은 빈칸으로 보이지 않도록 가운뎃점(·)으로 표시


# %% [markdown]
# **관찰 포인트**
# - `tokens` 는 **글자 단위**로 잘라져 있고, 공백도 하나의 토큰으로 포함됩니다.
# - 한 개체명(예: "서울")은 여러 글자 토큰에 걸쳐 `B-LC`, `I-LC` 가 이어집니다.
# - 즉 라벨은 "원래 글자 단위"로 붙어 있으므로, 곧 BERT 의 **sub-word 토큰** 에
#   다시 맞춰 주는(align) 작업이 반드시 필요해질 것입니다 (→ 과제 3).

# %% [markdown]
# ---
# ## 과제 2. 라벨 매핑 사전 만들기
#
# 모델 학습과 추론에서는 라벨을 ↔ 정수 ID 로 자유롭게 변환할 수 있어야 합니다.
#
# **할 일**:
# - `raw_datasets["train"].features["ner_tags"].feature.names` 에서 라벨 리스트를 가져오세요.
# - `id2label`, `label2id` dict 를 만들고 라벨 개수(`num_labels`)를 출력하세요.
#
# **힌트**: 라벨 리스트를 직접 손으로 적지 마세요 — 데이터셋이 정의한 순서를 그대로 써야
# `ner_tags` 의 정수값과 일치합니다.

# %%
# 라벨 목록을 데이터셋 feature 에서 직접 가져온다

# %% [markdown]
# ---
# ## 과제 3. Tokenizer 로드와 sub-word label alignment
#
# ### 왜 alignment 가 필요한가?
# KLUE-NER 의 라벨은 **글자 단위**로 붙어 있습니다. 그런데 BERT 토크나이저는
# 단어를 **sub-word** 로 다시 쪼개므로, "원래 글자 라벨"을 "sub-word 토큰"에
# 맞춰 주는(align) 작업이 반드시 필요합니다.
#
# - 한 단어의 **첫 번째 sub-word** → 원래 라벨 부여
# - 같은 단어의 **이어지는 sub-word** → `-100` (loss 계산 제외)
# - **특수 토큰** (`[CLS]`, `[SEP]`, padding) → `-100`
#
# > `-100` 은 PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값이라,
# > 해당 위치는 loss 계산에서 자동으로 무시됩니다.
#
# **할 일**:
# - `klue/bert-base` 토크나이저를 로드하세요.
# - `tokenize_and_align_labels` 함수를 작성해 글자 단위 라벨을 sub-word 토큰에 맞추세요.
# - 첫 샘플로 정렬 결과를 출력해 어떤 토큰에 어떤 라벨이 붙는지 확인하세요.
#
# **힌트**:
# - `tokenizer(..., is_split_into_words=True)` → 이미 단어 단위로 잘려 있는 리스트를 입력으로 받음.
# - `tokenized.word_ids(batch_index=i)` → 각 sub-word 가 원래 몇 번째 단어에서 왔는지 알려 줌.

# %%
# klue/bert-base 사전학습 토크나이저를 로드한다

# %%
# 이어지는 sub-word 의 라벨 처리 방식을 선택하는 옵션
#   "ignore" : 이어지는 sub-word 를 -100 으로 (loss 제외) — 가장 일반적
#   "same"   : 이어지는 sub-word 에도 첫 sub-word 와 같은 라벨 부여
def tokenize_and_align_labels(examples):
    # tokens 가 이미 분리된 리스트이므로 is_split_into_words=True 로 지정
        # word_ids(): 각 sub-word 토큰이 원래 몇 번째 단어에서 왔는지 알려준다
                # 특수 토큰([CLS], [SEP], padding) → loss 무시
                # 단어의 첫 sub-word → 원래 라벨 부여
                # 같은 단어의 이어지는 sub-word → 옵션에 따라 처리


# %%
# 변환 전/후를 직접 비교해 alignment 동작을 눈으로 확인한다

# %% [markdown]
# **관찰 포인트 — alignment 가 만드는 결과**
# - `[CLS]`, `[SEP]` 같은 특수 토큰은 모두 `-100 (무시)` 로 표시됩니다.
# - 한 글자가 여러 sub-word 로 쪼개진 경우, **첫 sub-word 에만** 원래 라벨이 붙고
#   뒷부분은 `-100` 으로 마스킹됩니다 → loss 가 한 번만 계산됩니다.
# - 이 덕분에 글자 단위 라벨과 sub-word 토큰 사이의 불일치가 자연스럽게 해소됩니다.

# %% [markdown]
# ---
# ## 과제 4. 전체 데이터셋에 토큰화·정렬 일괄 적용
#
# **할 일**:
# - `raw_datasets.map(tokenize_and_align_labels, batched=True, remove_columns=...)` 로
#   전체 데이터셋을 변환하세요.
# - 원본 컬럼(`tokens`, `ner_tags`, `sentence`)은 학습에 불필요하므로 제거합니다.

# %%
# map() 으로 전체 데이터셋에 alignment 를 일괄 적용한다
# remove_columns: 원본 컬럼은 학습에 불필요하므로 제거

# %% [markdown]
# ---
# ## 과제 5. Token Classification 모델 로드
#
# > **140 감성분석과의 차이점**:
# > 140 은 `AutoModelForSequenceClassification` (문장 1개 → 라벨 1개) 였지만,
# > 145 는 `AutoModelForTokenClassification` (토큰마다 라벨 1개) 을 사용합니다.
# > classifier head 가 `[CLS]` 한 자리가 아니라 **모든 토큰 위치**에 적용됩니다.
#
# **할 일**:
# - `AutoModelForTokenClassification.from_pretrained(...)` 으로 사전학습 모델 위에
#   token classification head 를 새로 붙이세요. `num_labels`, `id2label`, `label2id`
#   3가지 인자를 모두 넘겨 줘야 추론 시 라벨명을 그대로 출력할 수 있습니다.
# - `model.classifier` 를 출력해 head 의 입출력 차원을 확인하세요.

# %%
# token classification 용 head 가 붙은 모델을 로드한다 (head 는 새로 초기화됨)
# classifier head 출력: (hidden_size=768) → (num_labels=13) 선형 변환

# %% [markdown]
# **관찰 포인트**
# - classifier head 는 `Linear(768, 13)` 한 줄로 정의됩니다 (hidden_size → num_labels).
# - 사전학습 가중치는 그대로 사용하고, head 만 무작위 초기화되어 fine-tuning 으로 학습됩니다.

# %% [markdown]
# ---
# ## 과제 6. Trainer 구성 및 fine-tuning
#
# **할 일**:
# - `DataCollatorForTokenClassification` 으로 동적 패딩 콜레이터를 만드세요.
# - `seqeval` 기반 `compute_metrics` 함수를 작성하세요 (precision / recall / f1 / accuracy).
# - `TrainingArguments` 와 `Trainer` 를 구성해 학습을 실행하세요.
#
# **힌트**:
# - `seqeval` 은 BIO 태그 시퀀스를 **entity 단위**로 묶어 점수를 계산합니다.
#   단순 토큰 정확도가 아니라 "이순신 = 인물명 1개를 맞췄는가?" 를 판단합니다.
# - `-100` 으로 마스킹된 위치는 정답/예측에서 모두 빼고 점수를 계산해야 합니다.

# %%
# DataCollator: 배치 안에서 input_ids 와 labels 를 같은 길이로 동적 패딩한다
# (labels 의 패딩 자리는 자동으로 -100 으로 채워져 loss 에서 제외된다)

# %%
# seqeval 기반 평가 지표 함수를 정의한다
def compute_metrics(eval_preds):
    # -100 으로 마스킹된 위치(특수 토큰·이어지는 sub-word)는 제외하고
    # 정수 id 를 다시 문자열 BIO 태그로 변환한다

# %%
# (선택) 실행 시간이 오래 걸릴 경우 데이터 크기를 줄여 본다
# 전체 데이터로 학습하려면 아래 두 줄로 다시 덮어 쓴다

# %%
# 학습 하이퍼파라미터를 설정한다
# Trainer 객체를 생성한다

# %%
# 모델 fine-tuning 을 실행한다 (GPU 기준 수십 분 소요될 수 있음)
# 소요 시간을 시:분:초 형태로 보기 좋게 출력한다

# %% [markdown]
# ---
# ## 과제 7. 평가 및 entity 별 성능 분석
#
# **할 일**:
# - `trainer.evaluate()` 로 검증셋 종합 점수를 출력하세요.
# - `trainer.predict()` 결과를 BIO 태그 시퀀스로 복원해 `seqeval` 의
#   `classification_report` 로 **entity 유형별** 정밀도/재현율/F1 을 확인하세요.

# %%
# 검증셋 전체에 대한 종합 성능 지표를 출력한다

# %%
# entity 유형별(PS/LC/OG/DT/TI/QT) 정밀도·재현율·F1 을 상세 출력한다

# %% [markdown]
# **관찰 포인트**
# - PS(인물)·LC(장소)·OG(기관) 처럼 학습 샘플이 많은 유형은 보통 F1 이 높게 나옵니다.
# - DT(날짜)·TI(시간)·QT(수량) 은 표현 변이가 많아 점수가 상대적으로 낮을 수 있습니다.
# - `accuracy` 만 보면 90% 가 넘어 보여도, entity 단위 F1 은 그보다 훨씬 낮을 수 있습니다.
#   → NER 에서는 "토큰 정확도" 가 아니라 "entity F1" 이 더 의미 있는 지표입니다.

# %% [markdown]
# ---
# ## 과제 8. Fine-tuned 모델로 추론
#
# `pipeline("ner", ...)` 에 `aggregation_strategy="simple"` 을 주면
# 같은 개체에 속한 연속 토큰을 하나의 entity 로 묶어서 보여줍니다.
#
# **할 일**:
# - fine-tuned 모델로 NER `pipeline` 을 만드세요.
# - 한국어 예문 3개에 대해 인식된 개체와 신뢰도를 출력하세요.

# %%
# fine-tuned 모델로 NER 파이프라인을 구성한다

# %%
# entity_group 코드(PS/LC/OG/DT/TI/QT)를 한국어 명칭으로 바꿔주는 표
# 한국어 예문 3개에 대해 개체명 인식 결과를 읽기 쉽게 출력한다
        # 인식된 개체를 한 줄에 하나씩 [유형] 단어 (신뢰도) 형태로 출력

# %% [markdown]
# **관찰 포인트**
# - `aggregation_strategy="simple"` 덕분에 `[B-LC, I-LC, I-LC]` 가 자동으로
#   하나의 "장소" entity 로 묶여 반환됩니다.
# - 학습에 본 적 없는 신조어(예: "OpenAI", "샘 알트만") 도 BERT 의 sub-word
#   토크나이저 덕에 어느 정도 인식이 가능합니다 — 다만 신뢰도는 낮을 수 있습니다.

# %% [markdown]
# ---
# ## 과제 9. 모델 저장
#
# fine-tuned 모델과 토크나이저를 함께 저장하면 나중에
# `pipeline("ner", model="./klue-ner-bert-base")` 한 줄로 바로 재사용할 수 있습니다.
#
# **할 일**:
# - 모델과 토크나이저를 `./klue-ner-bert-base` 폴더에 저장하세요.

# %%
# fine-tuned 모델과 토크나이저를 같은 폴더에 저장한다

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 도구 | NER 고유 포인트 |
# |------|-----------|------------------|
# | ① 데이터 로드 | `load_dataset("klue", "ner")` | 라벨이 글자 단위 BIO 태그 |
# | ② 토큰화 + 정렬 | `tokenize_and_align_labels` + `word_ids()` | sub-word alignment, `-100` 마스킹 |
# | ③ fine-tuning | `AutoModelForTokenClassification` + `Trainer` | 모든 토큰에 classifier 적용 |
# | ④ 추론 | `pipeline("ner", aggregation_strategy="simple")` | 연속 토큰을 entity 로 병합 |
#
# **핵심 메시지**:
# - NER 은 "문장 → 라벨" 이 아니라 "토큰마다 라벨" 을 예측하는 **token classification** 문제입니다.
# - BERT 의 sub-word 토큰과 원본 글자 단위 라벨을 맞추는 **alignment** 가 NER fine-tuning 의 핵심 트릭입니다.
# - 평가는 토큰 정확도가 아닌 **entity 단위 F1** (`seqeval`) 로 보는 것이 의미 있습니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `LABEL_ALL_SUBTOKENS` 를 `"same"` 으로 바꿔 재학습하고 F1 이 어떻게 달라지는지
#    비교해 보세요 (이어지는 sub-word 에도 같은 라벨을 부여하는 변형).
# 2. `num_train_epochs` 를 5~10 으로 늘리고, 학습 곡선이 어디서부터 과적합되는지
#    검증 F1 을 epoch 별로 살펴보세요.
# 3. `examples` 리스트에 본인이 직접 만든 한국어 문장(특히 신조어·고유명사 포함)을
#    추가해 모델이 어떤 entity 는 잘 잡고 어떤 entity 에서 실패하는지 분석하세요.
# 4. 저장한 모델을 새 셀에서 `pipeline("ner", model="./klue-ner-bert-base",
#    aggregation_strategy="simple")` 으로 다시 로드해, 학습 직후의 파이프라인과
#    동일한 결과가 나오는지 확인하세요.

# %%
