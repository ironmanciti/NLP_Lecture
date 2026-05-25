# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PEFT LoRA 튜토리얼: 한국어 QA Fine-tuning 실습
#
# **데이터**: KorQuAD 1.0 (한국어 SQuAD)  
# **모델**: Qwen/Qwen2.5-0.5B (Alibaba Qwen 시리즈, 5억 파라미터)
#
# 이 노트북에서는 다음을 실습합니다:
# 1. 다국어 GPT 모델에 LoRA adapter 주입
# 2. KorQuAD 데이터로 QA fine-tuning 수행
# 3. 학습 전/후 한국어 QA 성능 비교 (정성 + 정량 평가)
# 4. 모델 저장 및 로드
#
# > ⚙️ Colab에서 **GPU 런타임**을 선택하세요: 런타임 → 런타임 유형 변경 → T4 GPU

# %% [markdown]
# ## 0. 환경 설정

# %%
# 필요한 라이브러리 설치

# %%
# GPU 확인

# %% [markdown]
# ## 1. LoRA Configuration 설정
#
# | 파라미터 | 값 | 설명 |
# |---------|---|------|
# | `r` | 32 | Low-rank 행렬의 rank |
# | `lora_alpha` | 32 | Scaling factor |
# | `target_modules` | all-linear | 모든 linear layer에 LoRA 적용 |
# | `lora_dropout` | 0.05 | 과적합 방지 |
# | `task_type` | CAUSAL_LM | 다음 토큰 예측 (생성 태스크) |

# %%

# %% [markdown]
# ## 2. Base Model 로드
#
# **Qwen/Qwen2.5-0.5B**  
# - Alibaba Qwen 시리즈의 경량 모델 (5억 파라미터)  
# - 다국어 대규모 코퍼스로 사전학습 (한국어 포함)  
# - 0.5B라 Colab T4에서 빠르게 학습 가능 → LoRA 효과를 체감하기 좋음  
# - HuggingFace: https://huggingface.co/Qwen/Qwen2.5-0.5B

# %%
# 토크나이저 로드
# Base 모델 로드 (fp16으로 메모리 절약)
# 패딩 토큰 설정

# %% [markdown]
# ## 3. PeftModel 생성
#
# `get_peft_model()`로 base model에 LoRA adapter를 주입합니다.  
# 원래 weights는 freeze, LoRA A/B 행렬만 학습됩니다.

# %%

# %%
# LoRA가 적용된 레이어 확인

# %% [markdown]
# ## 4. 한국어 QA 학습 데이터 준비
#
# ### 데이터셋: KorQuAD 1.0 (Korean Question Answering Dataset)
#
# SQuAD v1.0의 한국어 버전으로, 한국어 Wikipedia 문단에서 질문에 대한 답을 찾는 QA 데이터셋입니다.  
# 학습 데이터 60,407개, 검증 데이터 5,774개로 구성되어 있습니다.
#
# ```
# 문맥: {한국어 Wikipedia 문단}
# 질문: {질문}
# 답변: {정답}
# ```
#
# - 공식 사이트: https://korquad.github.io/
# - HuggingFace: https://huggingface.co/datasets/KorQuAD/squad_kor_v1

# %%
# KorQuAD 1.0 데이터 로드
# 예시 확인

# %%
# Colab T4에서 실습 가능한 크기로 샘플링

# %% [markdown]
# ### 데이터 전처리
#
# QA를 Causal LM 포맷으로 변환합니다.  
# 모델이 `답변:` 이후 부분만 생성하도록 **labels masking**을 적용합니다.

# %%
def format_qa(example):
    # 한국어 프롬프트 포맷
# 포맷 적용


# %% [markdown]
# PyTorch의 CrossEntropyLoss에서 ignore_index의 기본값이 -100입니다.
# label이 -100인 위치는 loss 계산에서 완전히 제외됩니다. 즉, 그 토큰은 "맞추든 틀리든 상관없다"는 의미입니다.

# %%
def tokenize_fn(examples):
        # 프롬프트 부분은 loss 계산에서 제외 (-100)
        # 패딩 부분도 제외
# 토크나이징 적용


# %% [markdown]
# ## 5. 학습 전 성능 확인 (baseline)
#
# 학습 전에 모델이 한국어 QA를 어떻게 하는지 확인합니다.  
# **정성 평가**(생성 텍스트 비교)와 **정량 평가**(Perplexity)를 모두 수행합니다.

# %%
def generate_answer(model, question, context, max_new_tokens=50):
    # "답변:" 이후 부분만 추출
# 테스트 샘플 5개
# ── 학습 전 생성 결과 저장 ──

# %% [markdown]
# ## 6. Trainer 준비 & 학습 전 Perplexity 측정
#
# 학습 전에 eval perplexity를 먼저 측정해 둡니다.  
# 학습 후와 비교하면 LoRA의 효과를 **수치로** 확인할 수 있습니다.

# %%

# %% [markdown]
# **Perplexity = 모델이 다음 토큰을 예측할 때 느끼는 "혼란도"**  
#
# Perplexity가 100이면 → 모델이 매 토큰마다 "100개 후보 중 하나"를 찍는 수준의 불확실성을 느낀다는 뜻  
# - 1 : 다음 토큰을 100% 확신, 10 : 꽤 잘 알고 있음, 100 : 많이 헷갈림

# %%
# ── 학습 전 Perplexity 측정 ──

# %% [markdown]
# ## 7. LoRA Fine-tuning 실행

# %%

# %% [markdown]
# ## 8. 학습 전/후 Perplexity 비교
#
# Perplexity가 낮을수록 모델이 정답 토큰을 잘 예측한다는 의미입니다.

# %%
# ── 학습 후 Perplexity 측정 ──

# %% [markdown]
# ### 8-2. Training Loss Curve

# %%

# %% [markdown]
# ### 8-3. 학습 전/후 생성 결과 비교
#
# 동일한 테스트 샘플로 학습 전/후를 나란히 비교합니다.

    # %%
    # 정답 포함 여부 체크

# %% [markdown]
# ## 9. 모델 저장 & 로드

# %%
# adapter 저장
# 저장된 파일 확인

# %%
# 저장된 adapter 다시 로드
# 로드된 모델로 테스트

# %% [markdown]
# ## 10. Merge & Unload (선택)
#
# 서빙 시 adapter를 base에 합쳐서 추론 속도를 개선할 수 있습니다.
#
# > ⚠️ merge 후에는 adapter를 분리할 수 없습니다.

# %%

# %% [markdown]
# ## 정리
#
# | 항목 | 값 |
# |------|---|
# | Base model | Qwen/Qwen2.5-0.5B (5억 파라미터) |
# | 데이터셋 | KorQuAD 1.0 (한국어 QA, 5,000개 사용) |
# | LoRA rank | 32 |
# | Target modules | all-linear |
# | 학습 에폭 | 3 |
# | Adapter 크기 | 수십 MB (base model ~1GB 대비 극소) |
#
# ### 더 나은 결과를 위한 팁
# - `TRAIN_SIZE`를 늘리기 (전체 60K 사용 시 성능 대폭 향상)
# - `num_train_epochs`를 5~10으로 늘리기
# - `r=64`로 rank 올리기 (메모리 허용 시)
# - 더 큰 base model 사용 (예: Qwen2.5-1.5B, Qwen2.5-3B)
#
# ### 참고 링크
# - [KorQuAD 공식](https://korquad.github.io/)
# - [PEFT 공식 문서](https://huggingface.co/docs/peft/en/index)
# - [LoRA 개발자 가이드](https://huggingface.co/docs/peft/en/developer_guides/lora)
