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
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 145. 실습 - HuggingFace NER Fine Tuning (KLUE-NER)
#
# ## 학습 목표
# `klue/bert-base`(https://huggingface.co/klue/bert-base) 사전학습 모델을 **개체명 인식(NER, Named Entity Recognition)** task로
# fine-tuning 합니다. KLUE-NER 데이터셋을 사용해 한국어 문장에서 인물·장소·기관 등을
# 자동으로 찾아내는 모델을 만듭니다.
#
# ## 140 (감성분석 / 문장 분류) 실습과의 핵심 차이
# | 항목 | 140. 감성분석 (Sequence Classification) | 145. NER (Token Classification) |
# |------|----------------------------------------|----------------------------------|
# | 모델 클래스 | `AutoModelForSequenceClassification` | `AutoModelForTokenClassification` |
# | 예측 단위 | 문장 1개 → 라벨 1개 (`[CLS]` 토큰만 사용) | 토큰마다 라벨 1개 (**모든 토큰** 사용) |
# | 라벨 형태 | 문장 단위 정수 (긍정/부정) | 토큰 단위 BIO 태그 시퀀스 |
#
# ## BIO 태그란?
# NER 의 라벨은 **BIO 태깅** 방식으로 표현합니다. 문장의 토큰마다 태그를 하나씩 붙이며,
# 태그는 세 가지 접두사로 구성됩니다.
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
# **B- 와 I- 를 구분하는 이유**: 인접한 두 개체를 구별하기 위해서입니다.
# `"서울 부산"` 의 경우 `B-LC B-LC` 면 별개의 장소 2개, `B-LC I-LC` 면 하나의 지명입니다.
# 즉 B- 가 "여기서 새 개체가 시작된다"는 경계 신호 역할을 합니다.
#
# **13개 라벨 계산**: KLUE-NER 은 6개 개체 유형(PS LC OG DT TI QT)을 다루므로
# `6개 유형 × 2(B-, I-) + O 1개 = 13개` 라벨이 됩니다. 이것이 뒤에서 `num_labels=13` 인 이유입니다.
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
# ## 2. 데이터셋 로드 및 탐색
#
# KLUE-NER 은 한국어 NER 벤치마크입니다.
# - `tokens` : **글자(character) 단위**로 분리된 리스트
# - `ner_tags` : 각 글자에 대응하는 BIO 태그 (정수 인코딩)
# - 6개 개체 유형: PS(인물) LC(장소) OG(기관) DT(날짜) TI(시간) QT(수량)

# %%
# KLUE-NER 데이터셋의 train / validation split 을 로드한다

# %%
# 학습 데이터 샘플 1~2개를 출력해 구조를 눈으로 확인한다
# 토큰(글자) · ner_tags(숫자) · 라벨명(알파벳)을 세로로 정렬해 한눈에 보이게 한다
# ner_tags 의 정수값 → 라벨명(B-LC, I-LC, O 등) 변환표
def disp_width(s):
def pad(s, width):
    # 토큰을 12개씩 끊어서, 토큰 / 숫자 / 라벨 3줄을 칸 맞춰 출력
        # 공백 토큰은 빈칸으로 보이지 않도록 가운뎃점(·)으로 표시
        # 각 칸의 폭 = (토큰, 숫자, 라벨) 중 가장 넓은 값


# %%
# 라벨 목록을 데이터셋 feature 에서 직접 가져온다
# 주의: ner_tags 의 정수값은 이 feature 의 순서로 인코딩되어 있으므로
#       라벨 리스트를 임의로 하드코딩하면 안 되고 반드시 feature 에서 가져와야 한다

# %% [markdown]
# ---
# ## 3. Tokenizer 준비 및 sub-word label alignment
#
# ### 왜 alignment 가 필요한가?
# KLUE-NER 의 라벨은 **글자 단위**로 붙어 있습니다. 그런데 BERT 토크나이저는
# 단어를 **sub-word** 로 다시 쪼갭니다. 따라서 "원래 글자 라벨"을
# "sub-word 토큰"에 다시 맞춰 주는(align) 작업이 반드시 필요합니다.
#
# - 한 단어의 **첫 번째 sub-word** → 원래 라벨 부여
# - 같은 단어의 **이어지는 sub-word** → `-100` (loss 계산 제외) 또는 동일 라벨
# - **특수 토큰** (`[CLS]`, `[SEP]`, padding) → `-100`
#
# > `-100` 은 PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값이라,
# > 해당 위치는 loss 계산에서 자동으로 무시됩니다.

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

# %%
# map() 으로 전체 데이터셋에 alignment 를 일괄 적용한다
# remove_columns: 원본 컬럼(tokens, ner_tags, sentence)은 학습에 불필요하므로 제거

# %% [markdown]
# ---
# ## 4. 모델 로드
#
# > **140 감성분석과의 차이점**:
# > 140 은 `AutoModelForSequenceClassification` (문장 1개 → 라벨 1개) 였지만,
# > 145 는 `AutoModelForTokenClassification` (토큰마다 라벨 1개) 을 사용합니다.
# > classifier head 가 `[CLS]` 한 자리가 아니라 **모든 토큰 위치**에 적용됩니다.

# %%
# token classification 용 head 가 붙은 모델을 로드한다 (head 는 새로 초기화됨)
# classifier head 출력: (hidden_size=768) → (num_labels=13) 선형 변환

# %% [markdown]
# ---
# ## 5. Trainer 구성 및 학습

# %%
# DataCollator: 배치 안에서 input_ids 와 labels 를 같은 길이로 동적 패딩한다
# (labels 의 패딩 자리는 자동으로 -100 으로 채워져 loss 에서 제외된다)

# %%
# seqeval 기반 평가 지표 함수를 정의한다
# seqeval 은 BIO 태그 시퀀스를 entity 단위로 묶어 정밀도/재현율/F1 을 계산한다
def compute_metrics(eval_preds):
    # -100 으로 마스킹된 위치(특수 토큰·이어지는 sub-word)는 제외하고
    # 정수 id 를 다시 문자열 BIO 태그로 변환한다

# %%
# (선택) 실행 시간이 오래 걸릴 경우 데이터 크기를 줄인다

# %%
# 학습 하이퍼파라미터를 설정한다
# Trainer 객체를 생성한다

# %%
# 모델 fine-tuning 을 실행한다 (GPU 기준 수십 분 소요될 수 있음)
# 소요 시간을 시:분:초 형태로 보기 좋게 출력한다

# %% [markdown]
# ---
# ## 6. 평가 및 entity 별 성능

# %%
# 검증셋 전체에 대한 종합 성능 지표를 출력한다

# %%
# entity 유형별(PS/LC/OG/DT/TI/QT) 정밀도·재현율·F1 을 상세 출력한다

# %% [markdown]
# ---
# ## 7. Fine-tuned 모델로 추론
#
# `pipeline("ner", ...)` 에 `aggregation_strategy="simple"` 을 주면
# 같은 개체에 속한 연속 토큰을 하나의 entity 로 묶어서 보여줍니다.

# %%
# fine-tuned 모델로 NER 파이프라인을 구성한다

# %%
# entity_group 코드(PS/LC/OG/DT/TI/QT)를 한국어 명칭으로 바꿔주는 표
# 한국어 예문 3개에 대해 개체명 인식 결과를 읽기 쉽게 출력한다
        # 인식된 개체를 한 줄에 하나씩 [유형] 단어 (신뢰도) 형태로 출력

# %% [markdown]
# ---
# ## 8. 모델 저장
#
# fine-tuned 모델과 토크나이저를 함께 저장하면 나중에
# `pipeline("ner", model="./klue-ner-bert-base")` 로 바로 재사용할 수 있습니다.

# %%
# fine-tuned 모델과 토크나이저를 같은 폴더에 저장한다

# %% [markdown]
# ---
# ## 정리
#
# | 단계 | 핵심 도구 | NER 고유 포인트 |
# |------|-----------|------------------|
# | ① 데이터 로드 | `load_dataset("klue", "ner")` | 라벨이 글자 단위 BIO 태그 |
# | ② 토큰화 + 정렬 | `tokenize_and_align_labels` + `word_ids()` | sub-word alignment, `-100` 마스킹 |
# | ③ fine-tuning | `AutoModelForTokenClassification` + `Trainer` | 모든 토큰에 classifier 적용 |
# | ④ 추론 | `pipeline("ner", aggregation_strategy="simple")` | 연속 토큰을 entity 로 병합 |
#

# %%
