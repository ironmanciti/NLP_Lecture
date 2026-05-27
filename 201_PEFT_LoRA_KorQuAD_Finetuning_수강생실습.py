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
# # 200. 수강생 실습 — PEFT LoRA 로 한국어 QA Fine-tuning (KorQuAD)
#
# ## 학습 목표
# 사전학습된 Qwen2.5-0.5B 모델에 **LoRA adapter** 만 얹어서, KorQuAD (한국어 QA)
# 데이터로 한국어 질의응답을 잘하는 모델로 fine-tuning 합니다.
# 전체 모델을 다시 학습시키지 않고도 한국어 QA 성능을 끌어올릴 수 있다는 점을
# 직접 측정해 보는 실습입니다.
#
# ## LoRA 가 풀고 싶은 문제 — full fine-tuning 의 한계
# - LLM 의 모든 가중치를 직접 학습시키려면 **VRAM·디스크·시간** 이 어마어마하게 듭니다.
# - LoRA 는 원본 가중치를 **얼려 두고**, 각 linear 층 옆에 작은 **low-rank 행렬 두 개
#   (A, B)** 만 끼워 넣어 그쪽만 학습합니다.
# - 학습 가능한 파라미터가 보통 **원래의 1% 미만**으로 줄어, Colab 의 T4 한 장으로도
#   대규모 모델 fine-tuning 이 가능해집니다.
# - 학습 결과는 **수십 MB 짜리 adapter 파일** 만 저장하면 됩니다.
#
# ## 이번 실습의 비교 포인트
# | | 학습 전 (base model) | 학습 후 (LoRA fine-tuned) |
# |---|----------------------|---------------------------|
# | 정성 평가 | 생성 결과 5개 비교 | 같은 5개에서 정답 포함 여부 |
# | 정량 평가 | Perplexity | 같은 eval set 에서 Perplexity |
#
# ## 워크플로우 4단계
# ```
#   ① LoRA 설정 + 모델 로드  →  ② 데이터 준비  →  ③ Trainer fine-tuning  →  ④ 비교/저장
# ```
#
# > **실행 환경**: Colab GPU 권장 (T4 면 충분). CPU 로는 매우 느립니다.

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.

# %%
# !pip install -q peft accelerate datasets
# !pip install -q --upgrade torchao

# %% [markdown]
# ---
# ## 1. 환경 확인

# %%
import torch

# GPU 확인
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용 디바이스: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# %% [markdown]
# ---
# ## 과제 1. LoRA Configuration 설정
#
# 학습할 LoRA adapter 의 형태를 먼저 정의합니다. 핵심 하이퍼파라미터는 다음과 같습니다.
#
# | 파라미터 | 값 | 설명 |
# |---------|---|------|
# | `r` | 32 | Low-rank 행렬의 rank (작을수록 가벼움, 너무 작으면 표현력 부족) |
# | `lora_alpha` | 32 | Scaling factor — 실효 학습률에 비례하는 효과 |
# | `target_modules` | `"all-linear"` | 모든 linear layer 에 LoRA 적용 |
# | `lora_dropout` | 0.05 | 과적합 방지 |
# | `task_type` | `CAUSAL_LM` | 다음 토큰 예측 (생성 태스크) |
#
# **할 일**:
# - `peft.LoraConfig` 로 위 표대로 설정 객체를 만드세요.
# - 출력해 어떤 필드가 들어 있는지 확인하세요.
#
# **힌트**: `target_modules="all-linear"` 는 peft 가 모델에서 모든 `nn.Linear` 를 찾아
# 자동으로 LoRA 를 끼워 넣어 줍니다. 일일이 모듈 이름을 적지 않아도 됩니다.

# %%
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=32,                                # rank: low-rank 행렬의 차원
    target_modules="all-linear",         # 모든 linear layer 에 LoRA 적용
    task_type=TaskType.CAUSAL_LM,        # Causal Language Modeling
    lora_alpha=32,                       # scaling factor
    lora_dropout=0.05                    # dropout
)

print(lora_config)

# %% [markdown]
# ---
# ## 과제 2. Base 모델과 토크나이저 로드
#
# **Qwen/Qwen2.5-0.5B**
# - Alibaba Qwen 시리즈의 경량 모델 (5억 파라미터)
# - 다국어 대규모 코퍼스로 사전학습 (한국어 포함)
# - 0.5B 라 Colab T4 에서 빠르게 학습 가능 → LoRA 효과를 체감하기 좋음
# - HuggingFace: https://huggingface.co/Qwen/Qwen2.5-0.5B
#
# **할 일**:
# - `AutoTokenizer` 와 `AutoModelForCausalLM` 으로 모델/토크나이저를 로드하세요.
# - `torch_dtype=torch.float16, device_map="auto"` 로 메모리를 절약하세요.
# - `pad_token` 이 없으면 `eos_token` 으로 채워 넣으세요 (생성 모델은 패딩 토큰이
#   따로 없는 경우가 많습니다).
#
# **힌트**: `pad_token` 을 설정하지 않으면 학습 시 패딩 자리에서 NaN/경고가 발생합니다.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Base 모델 로드 (fp16 으로 메모리 절약)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 패딩 토큰 설정 (생성 모델은 보통 pad_token 이 없음)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

print(f"모델: {model_name}")
print(f"파라미터 수: {model.num_parameters():,}")
print(f"토크나이저 vocab 크기: {len(tokenizer):,}")

# %% [markdown]
# ---
# ## 과제 3. LoRA adapter 주입 (`get_peft_model`)
#
# `get_peft_model()` 한 줄이면 base model 의 모든 linear 층 옆에 LoRA A/B 행렬이
# 자동으로 끼워집니다. 원본 가중치는 모두 `requires_grad=False` 로 얼려지고,
# **LoRA A/B 행렬만** `requires_grad=True` 가 됩니다.
#
# **할 일**:
# - `get_peft_model(model, lora_config)` 로 LoRA adapter 가 붙은 모델을 만드세요.
# - `print_trainable_parameters()` 로 학습 가능한 파라미터 비율을 확인하세요.
# - `named_parameters()` 에서 `requires_grad=True` 인 항목 일부를 출력해
#   실제로 어느 위치에 LoRA 가 붙었는지 눈으로 확인하세요.

# %%
import torch
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# %%
# LoRA 가 적용된 레이어 확인
trainable_params = [(name, param.shape) for name, param in lora_model.named_parameters() if param.requires_grad]
print(f"학습 가능한 파라미터 그룹 수: {len(trainable_params)}\n")
for name, shape in trainable_params[:10]:  # 처음 10개만
    print(f"  ✅ {name} | {shape}")
if len(trainable_params) > 10:
    print(f"  ... 외 {len(trainable_params) - 10}개")

# %% [markdown]
# **관찰 포인트**
# - `print_trainable_parameters()` 출력에서 **trainable / total 비율** 이
#   보통 **1% 미만** 으로 나옵니다. 이것이 LoRA 의 핵심 효율입니다.
# - 학습 가능한 파라미터 이름은 `...lora_A.default.weight`, `...lora_B.default.weight`
#   처럼 끝납니다 — base 모델의 각 linear 층 옆에 한 쌍씩 붙는 구조입니다.

# %% [markdown]
# ---
# ## 과제 4. KorQuAD 데이터셋 로드 및 전처리
#
# **데이터셋: KorQuAD 1.0 (Korean Question Answering Dataset)**
# - SQuAD v1.0 의 한국어 버전. 한국어 Wikipedia 문단에서 정답 추출.
# - 학습 60,407 / 검증 5,774 문항. 이번 실습에서는 일부만 샘플링해 사용합니다.
#
# ```
# 문맥: {한국어 Wikipedia 문단}
# 질문: {질문}
# 답변: {정답}
# ```
#
# **할 일**:
# - `load_dataset("KorQuAD/squad_kor_v1", split="train")` 으로 로드하세요.
# - 5,000 + 200 개를 무작위로 뽑아 학습/평가 split 으로 나누세요.
# - 각 예제를 `format_qa()` 로 `"문맥: ... \n질문: ... \n답변: ..."` 형식의
#   한 줄 텍스트로 변환하세요.
# - `tokenize_fn()` 에서 **프롬프트 부분 라벨을 -100 으로 마스킹** 하세요.
#   → 모델이 답변 부분에서만 loss 를 계산하도록 만드는 핵심 트릭입니다.
#
# **힌트**:
# - PyTorch `CrossEntropyLoss` 의 `ignore_index` 기본값이 `-100` 이므로
#   `-100` 인 위치는 자동으로 loss 에서 제외됩니다.
# - 패딩 자리도 함께 `-100` 으로 마스킹해야 패딩이 학습에 영향을 주지 않습니다.

# %%
from datasets import load_dataset

# KorQuAD 1.0 데이터 로드
dataset = load_dataset("KorQuAD/squad_kor_v1", split="train")
print(f"전체 학습 데이터 수: {len(dataset):,}")

# 예시 확인
for i in [0, 1000]:
    example = dataset[i]
    print(f"\n{'='*50}")
    print(f"제목: {example['title']}")
    print(f"문맥: {example['context'][:150]}...")
    print(f"질문: {example['question']}")
    print(f"답변: {example['answers']['text'][0]}")

# %%
# Colab T4 에서 실습 가능한 크기로 샘플링
TRAIN_SIZE = 5000
EVAL_SIZE = 200

small_dataset = dataset.shuffle(seed=42).select(range(TRAIN_SIZE + EVAL_SIZE))
train_dataset = small_dataset.select(range(TRAIN_SIZE))
eval_dataset = small_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

print(f"학습 데이터: {len(train_dataset)}개")
print(f"평가 데이터: {len(eval_dataset)}개")

# %% [markdown]
# ### 데이터 포맷팅
#
# QA 를 Causal LM 포맷의 평문 텍스트로 변환합니다.

# %%
MAX_LENGTH = 256


def format_qa(example):
    """KorQuAD 예제를 생성형 QA 텍스트로 변환"""
    context = example["context"][:300]  # context 길이 제한
    question = example["question"]
    answer = example["answers"]["text"][0]

    # 한국어 프롬프트 포맷
    prompt = f"문맥: {context}\n질문: {question}\n답변:"
    full_text = f"{prompt} {answer}{tokenizer.eos_token}"

    return {"prompt": prompt, "full_text": full_text}


# 포맷 적용
train_dataset = train_dataset.map(format_qa)
eval_dataset = eval_dataset.map(format_qa)

print("=== 변환된 예시 ===")
print(train_dataset[0]["full_text"][:400])

# %% [markdown]
# ### 토크나이징 + Labels Masking
#
# PyTorch 의 `CrossEntropyLoss` 에서 `ignore_index` 의 기본값이 `-100` 입니다.
# label 이 `-100` 인 위치는 loss 계산에서 완전히 제외됩니다.
# 즉, 그 토큰은 "맞추든 틀리든 상관없다" 는 의미입니다.

# %%
def tokenize_fn(examples):
    """토크나이징 + labels masking (프롬프트 부분은 -100 으로)"""
    model_inputs = tokenizer(
        examples["full_text"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )

    labels = []
    for i, prompt in enumerate(examples["prompt"]):
        prompt_len = len(tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)["input_ids"])
        label = model_inputs["input_ids"][i].copy()
        # 프롬프트 부분은 loss 계산에서 제외 (-100)
        label[:prompt_len] = [-100] * prompt_len
        # 패딩 부분도 제외
        label = [-100 if token == tokenizer.pad_token_id else l
                 for token, l in zip(model_inputs["input_ids"][i], label)]
        labels.append(label)

    model_inputs["labels"] = labels
    return model_inputs


# 토크나이징 적용
tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
tokenized_eval = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

print("토크나이징 완료")
print(f"  학습 데이터 컬럼: {tokenized_train.column_names}")
print(f"  입력 시퀀스 길이: {len(tokenized_train[0]['input_ids'])}")

# %% [markdown]
# **관찰 포인트**
# - 학습 데이터의 `labels` 에서 앞쪽은 모두 `-100` 이고, "답변:" 직후부터만 실제
#   토큰 id 값이 들어 있습니다. 이 마스킹 덕분에 모델은 "문맥과 질문을 베껴쓰는"
#   학습을 하지 않고 **답변만 생성** 하는 데 집중합니다.

# %% [markdown]
# ---
# ## 과제 5. 학습 전 (baseline) 성능 측정
#
# fine-tuning 의 효과를 수치/정성 모두로 확인하기 위해 **학습을 시작하기 전에 먼저**
# 모델의 현재 상태를 측정해 둡니다.
#
# **할 일**:
# - `generate_answer()` 함수를 정의해 문맥+질문을 입력하면 답변만 추출해 반환하도록 만드세요.
# - 평가 데이터 5개에 대해 학습 전 생성 결과를 `baseline_preds` 리스트에 저장하세요.
# - (과제 6에서 Trainer 를 구성한 뒤) `trainer.evaluate()` 로 학습 전 Perplexity 도 기록합니다.

# %%
def generate_answer(model, question, context, max_new_tokens=50):
    """주어진 문맥과 질문으로 답변 생성"""
    prompt = f"문맥: {context}\n질문: {question}\n답변:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id   # generate 시 pad 경고 방지
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # "답변:" 이후 부분만 추출
    answer = full_output.split("답변:")[-1].strip()
    return answer


# 테스트 샘플 5개
test_samples = eval_dataset.select(range(5))

# ── 학습 전 생성 결과 저장 ──
baseline_preds = []

print("=" * 60)
print("📋 학습 전 (baseline) 생성 결과")
print("=" * 60)

for i, sample in enumerate(test_samples):
    pred = generate_answer(lora_model, sample["question"], sample["context"][:300])
    gold = sample["answers"]["text"][0]
    baseline_preds.append(pred)
    print(f"\n[{i+1}] 질문: {sample['question']}")
    print(f"    정답: {gold}")
    print(f"    예측: {pred[:100]}")

# %% [markdown]
# **관찰 포인트**
# - 학습 전 base 모델은 한국어 QA 포맷에 익숙하지 않아 엉뚱하거나 장황한 답을 내놓기 쉽습니다.
# - "정답 키워드가 예측에 포함되어 있는가?" 정도가 가장 단순한 정성 평가 기준입니다.

# %% [markdown]
# ---
# ## 과제 6. Trainer 준비 & 학습 전 Perplexity 측정
#
# `TrainingArguments` 와 `Trainer` 를 구성합니다. Causal LM 학습이므로
# `DataCollatorForLanguageModeling(..., mlm=False)` 를 씁니다.
#
# **할 일**:
# - `TrainingArguments` 로 학습 하이퍼파라미터를 설정하세요.
# - `Trainer` 객체를 만들고, **학습을 시작하기 전에** `trainer.evaluate()` 로
#   기준선(Perplexity) 을 측정해 두세요.
#
# **힌트**:
# - `gradient_accumulation_steps=2` → 효과 배치 크기 = `8 × 2 = 16`. 메모리 한계에서
#   배치를 키운 효과를 내는 트릭입니다.
# - **Perplexity = exp(loss)** 입니다. loss 가 1 줄어들면 perplexity 는 약 2.7배 좋아집니다.
#
# > **Perplexity = 모델이 다음 토큰을 예측할 때 느끼는 "혼란도"**
# > - 1 : 다음 토큰을 100% 확신, 10 : 꽤 잘 알고 있음, 100 : 많이 헷갈림
# > - 100 이면 모델이 매 토큰마다 "100개 후보 중 하나" 를 찍는 수준의 불확실성을 느낀다는 뜻

# %%
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./lora-korquad-checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,        # effective batch = 8 * 2 = 16
    learning_rate=3e-4,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    remove_unused_columns=False,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal LM
)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_eval,
    data_collator=data_collator,
)

print("✅ Trainer 준비 완료")
print(f"  모델: {model_name}")
print(f"  학습 데이터: {len(tokenized_train)}개")
print(f"  평가 데이터: {len(tokenized_eval)}개")
print(f"  에폭: {training_args.num_train_epochs}")
print(f"  Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")

# %%
# ── 학습 전 Perplexity 측정 ──
baseline_eval = trainer.evaluate()
baseline_ppl = torch.exp(torch.tensor(baseline_eval['eval_loss'])).item()
print(f"📊 학습 전 Eval Loss: {baseline_eval['eval_loss']:.4f}")
print(f"📊 학습 전 Perplexity: {baseline_ppl:.2f}")

# %% [markdown]
# ---
# ## 과제 7. LoRA Fine-tuning 실행
#
# **할 일**:
# - `trainer.train()` 으로 LoRA fine-tuning 을 실행하세요.
# - 학습 종료 후 step 수, training loss, 소요 시간을 출력하세요.
#
# **힌트**: 학습 가능한 파라미터가 1% 미만이므로 메모리 사용량이 매우 작습니다.
# Colab T4 에서 5,000 샘플 × 3 epoch 기준 보통 수 분~십여 분이면 끝납니다.

# %%
print("학습을 시작합니다...\n")
train_result = trainer.train()

print(f"\n{'=' * 40}")
print(f"✅ 학습 완료!")
print(f"  Total steps: {train_result.global_step}")
print(f"  Training loss: {train_result.training_loss:.4f}")
print(f"  학습 시간: {train_result.metrics['train_runtime']:.1f}초")

# %% [markdown]
# ---
# ## 과제 8. 학습 전/후 비교 (Perplexity + Loss curve + 생성 결과)
#
# **할 일**:
# - 학습 후 다시 `trainer.evaluate()` 로 Perplexity 를 측정하고 baseline 과 비교하세요.
# - `trainer.state.log_history` 에서 training loss 를 뽑아 곡선으로 그려 보세요.
# - 과제 5 의 `baseline_preds` 와 새 예측을 같은 5개 샘플에 대해 나란히 출력하세요.
#
# **힌트**: 정답 키워드 포함 여부(`gold in pred`) 가 가장 단순한 자동 판정 기준입니다.

# %%
# ── 학습 후 Perplexity 측정 ──
lora_model.eval()
after_eval = trainer.evaluate()
after_ppl = torch.exp(torch.tensor(after_eval['eval_loss'])).item()

print("=" * 50)
print("📊 Perplexity 비교")
print("=" * 50)
print(f"  학습 전: {baseline_ppl:.2f}")
print(f"  학습 후: {after_ppl:.2f}")
print(f"  개선율:  {(baseline_ppl - after_ppl) / baseline_ppl * 100:.1f}%")
print()
print(f"  학습 전 Eval Loss: {baseline_eval['eval_loss']:.4f}")
print(f"  학습 후 Eval Loss: {after_eval['eval_loss']:.4f}")

# %% [markdown]
# ### Training Loss Curve

# %%
import matplotlib.pyplot as plt

logs = [l for l in trainer.state.log_history if 'loss' in l]
steps = [l['step'] for l in logs]
losses = [l['loss'] for l in logs]

plt.figure(figsize=(8, 4))
plt.plot(steps, losses, 'b-', linewidth=1.5)
plt.xlabel('Step')
plt.ylabel('Training Loss')
plt.title('LoRA Fine-tuning Loss Curve')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"초기 loss: {losses[0]:.4f} → 최종 loss: {losses[-1]:.4f}")

# %% [markdown]
# ### 학습 전/후 생성 결과 비교
#
# 동일한 테스트 샘플로 학습 전/후를 나란히 비교합니다.

# %%
print("=" * 70)
print("📋 학습 전/후 생성 결과 비교")
print("=" * 70)

for i, sample in enumerate(test_samples):
    pred_after = generate_answer(lora_model, sample["question"], sample["context"][:300])
    gold = sample["answers"]["text"][0]
    pred_before = baseline_preds[i]

    # 정답 포함 여부 체크
    before_ok = gold in pred_before
    after_ok = gold in pred_after

    print(f"\n[{i+1}] 질문: {sample['question']}")
    print(f"    정답:     {gold}")
    print(f"    학습 전:  {pred_before[:100]}  {'✅' if before_ok else '❌'}")
    print(f"    학습 후:  {pred_after[:100]}  {'✅' if after_ok else '❌'}")

# %% [markdown]
# **관찰 포인트**
# - Perplexity 가 학습 전 대비 크게(보통 수 배~수십 배) 감소하면 fine-tuning 이 잘 된 것입니다.
# - Loss curve 가 초기 빠른 감소 → 후반 완만한 감소 형태를 보이면 정상.
# - 학습 후 예측에서 정답 키워드 포함 비율이 baseline 보다 명확히 올라가야 의미가 있습니다.

# %% [markdown]
# ---
# ## 과제 9. 모델 저장 & 로드 & Merge
#
# LoRA 의 큰 장점 — adapter 파일만 저장하면 됩니다. base 모델(~1 GB)은 그대로 두고,
# 학습으로 만들어진 **A/B 행렬만 수십 MB** 로 저장됩니다.
#
# **할 일**:
# - `lora_model.save_pretrained(save_path)` 와 `tokenizer.save_pretrained(save_path)` 로 저장하세요.
# - 저장된 파일 목록과 전체 크기를 출력해 1 GB 미만임을 확인하세요.
# - `PeftConfig` + `PeftModel.from_pretrained` 으로 adapter 를 다시 로드해 동일한
#   추론 결과가 나오는지 확인하세요.
# - 마지막으로 `merge_and_unload()` 로 adapter 를 base 에 합쳐 단일 모델로 만들어 봅니다.
#
# > ⚠️ `merge_and_unload` 후에는 adapter 를 다시 분리할 수 없습니다.
# > 서빙 단계에서 "adapter 를 떼고 붙이는" 유연성을 잃는 대신 추론이 약간 빨라집니다.

# %%
# adapter 저장
save_path = "./qwen2.5-lora-korquad"
lora_model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# 저장된 파일 확인
import os

total_size = 0
print("저장된 파일:")
for f in sorted(os.listdir(save_path)):
    size = os.path.getsize(os.path.join(save_path, f))
    total_size += size
    print(f"  {f} ({size / 1024:.1f} KB)")
print(f"\n총 adapter 크기: {total_size / 1024 / 1024:.1f} MB")
print(f"(참고: base model 전체는 ~1 GB)")

# %%
from peft import PeftModel, PeftConfig

# 저장된 adapter 다시 로드
config = PeftConfig.from_pretrained(save_path)
print(f"Base model: {config.base_model_name_or_path}")

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
loaded_model = PeftModel.from_pretrained(base_model, save_path)
loaded_model.eval()

# 로드된 모델로 테스트
sample = eval_dataset[0]
pred = generate_answer(loaded_model, sample["question"], sample["context"][:300])
print(f"\n질문: {sample['question']}")
print(f"정답: {sample['answers']['text'][0]}")
print(f"예측: {pred}")
print(f"\n✅ 저장된 adapter 로드 & 추론 성공")

# %% [markdown]
# ### Merge & Unload (선택)
#
# 서빙 시 adapter 를 base 에 합쳐서 추론 속도를 개선할 수 있습니다.

# %%
merged_model = loaded_model.merge_and_unload()
print(f"Merged 모델 타입: {type(merged_model).__name__}")

pred = generate_answer(merged_model, sample["question"], sample["context"][:300])
print(f"\n질문: {sample['question']}")
print(f"예측: {pred}")
print(f"\n✅ Merged 모델 추론 성공")

# %% [markdown]
# **관찰 포인트**
# - 저장된 adapter 폴더는 base 모델 가중치를 포함하지 않으므로 매우 가볍습니다 (~수십 MB).
# - 같은 base 모델 위에 여러 개의 task-specific adapter 를 갈아 끼울 수 있다는 점이
#   LoRA 의 운영상 강점입니다 (예: 의료-QA adapter, 법률-QA adapter, ...).

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 도구 | LoRA 고유 포인트 |
# |------|-----------|------------------|
# | ① LoRA 설정 | `LoraConfig(r, target_modules, task_type)` | rank 와 적용 위치를 한 객체로 표현 |
# | ② adapter 주입 | `get_peft_model(model, lora_config)` | 원본 가중치 freeze, A/B 만 학습 |
# | ③ 데이터 전처리 | `format_qa` + 프롬프트 라벨 `-100` 마스킹 | 답변 부분에서만 loss 계산 |
# | ④ 학습 | `Trainer` (`mlm=False` collator) | 학습 파라미터 1% 미만 |
# | ⑤ 평가 | Perplexity + 생성 결과 비교 | 학습 전/후 차이를 수치로 확인 |
# | ⑥ 배포 | `save_pretrained` → `PeftModel.from_pretrained` → `merge_and_unload` | adapter 만 갈아 끼우는 운영 가능 |
#
# **핵심 메시지**:
# - LoRA 는 "원본 가중치를 얼리고 옆에 작은 행렬만 학습" 이라는 한 줄 아이디어로
#   거대 모델 fine-tuning 의 비용 장벽을 크게 낮춥니다.
# - 작은 데이터·짧은 epoch·작은 모델 조합이어도 perplexity 와 생성 결과로 분명한
#   개선을 측정할 수 있습니다.
# - 학습 결과는 base 모델과 분리된 **adapter 파일** 로 저장되어, 같은 base 위에 여러
#   task adapter 를 자유롭게 교체할 수 있습니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `TRAIN_SIZE` 를 `10000` 또는 전체(`60000`) 로 늘려 다시 학습하고 perplexity 가
#    얼마나 더 떨어지는지 확인하세요.
# 2. `LoraConfig` 의 `r` 을 `8`, `16`, `64` 로 바꿔 가며 학습량과 성능의 trade-off
#    (학습 시간, adapter 크기, perplexity) 를 비교하세요.
# 3. base model 을 `Qwen/Qwen2.5-1.5B` 또는 다른 한국어 친화 모델로 바꿔 같은 절차로
#    학습해 보고, 모델 크기가 학습 후 한국어 QA 품질에 미치는 영향을 살피세요.
# 4. 본인이 직접 만든 한국어 문맥/질문/정답 세트 3~5개로 `generate_answer` 를 호출해,
#    LoRA fine-tuning 으로 학습된 답변 스타일(짧고 추출형) 이 유지되는지 확인하세요.
#
# ### 참고 링크
# - [KorQuAD 공식](https://korquad.github.io/)
# - [PEFT 공식 문서](https://huggingface.co/docs/peft/en/index)
# - [LoRA 개발자 가이드](https://huggingface.co/docs/peft/en/developer_guides/lora)

# %%
