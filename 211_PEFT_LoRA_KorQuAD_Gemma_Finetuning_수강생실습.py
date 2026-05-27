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
# # 210. 수강생 실습 — PEFT LoRA 로 한국어 QA Fine-tuning (Gemma 3)
#
# ## 학습 목표
# [201 수강생실습](201_PEFT_LoRA_KorQuAD_Finetuning_수강생실습.py) 에서 Qwen2.5-0.5B 로
# 했던 한국어 QA LoRA fine-tuning 을, **Google Gemma 3 (1B)** 로 base 모델만 바꿔
# 그대로 재현해 봅니다. 같은 파이프라인이 다른 base 모델에서도 동작한다는 것을
# 직접 확인하는 비교 실습입니다.
#
# ## 200 (Qwen) 실습과의 핵심 차이
# | 항목 | 200. Qwen2.5-0.5B | **210. Gemma 3 1B** |
# |------|--------------------|---------------------|
# | Base 모델 | Qwen/Qwen2.5-0.5B (5억) | google/gemma-3-1b-it (10억) |
# | 사용 인증 | 불필요 | **gated 모델 — HF_TOKEN 필요** |
# | vocab 크기 | ~150K | **~262K** (logits 메모리 큼) |
# | `MAX_LENGTH` | 256 | **384** (Gemma 한국어 토큰이 잘게 쪼개짐) |
# | T4 메모리 대응 | micro batch 8, accum 2 | **micro batch 2, accum 8 + gradient checkpointing** |
# | adapter 파일 크기 | 수십 MB | 수십 MB (둘 다 base 대비 극소) |
#
# 두 노트북을 끝까지 돌려 본 뒤, **같은 5개 샘플에 대한 학습 전/후 결과**가
# Qwen vs Gemma 에서 어떻게 다른지 비교해 보는 것이 이 실습의 핵심 학습 포인트입니다.
#
# ## 워크플로우 4단계
# ```
#   ① 환경/인증 + LoRA 설정 + 모델 로드  →  ② 데이터 준비  →  ③ Trainer fine-tuning  →  ④ 비교/저장
# ```
#
# > **실행 환경**: Colab GPU (T4) 권장.
# > **사전 작업 — gated 모델 동의**: https://huggingface.co/google/gemma-3-1b-it 에서
# > 라이센스에 동의하고, HuggingFace 에서 발급받은 토큰을 `.env` 의 `HF_TOKEN` 또는
# > Colab Secrets 에 `HF_TOKEN` 으로 등록하세요.

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.
# Gemma 3 는 transformers 4.50+ 가 필요하므로 `-U` 로 업그레이드합니다.

# %%
# !pip install -q -U transformers peft accelerate datasets python-dotenv
# !pip install -q --upgrade torchao

# %% [markdown]
# ---
# ## 과제 1. HuggingFace 인증 (gated 모델 접근)
#
# Gemma 는 **gated 모델** 입니다. 라이센스 동의 + 토큰 인증을 모두 통과해야
# 가중치를 다운로드할 수 있습니다.
#
# **할 일**:
# - `.env` 에서 `HF_TOKEN` 을 불러오세요 (`python-dotenv` 사용).
# - `huggingface_hub.login(token=...)` 으로 로그인하세요.
# - GPU 정보를 출력해 학습 가능한 환경인지 확인하세요.
#
# **힌트**:
# - Colab 에서는 Secrets (🔑 자물쇠 아이콘) 에 `HF_TOKEN` 을 등록해 두면
#   `os.environ["HF_TOKEN"]` 로 그대로 읽을 수 있습니다.
# - 로컬에서는 프로젝트 루트의 `.env` 에 `HF_TOKEN=hf_...` 한 줄을 두면 됩니다.

# %%
import os

from dotenv import load_dotenv
load_dotenv()

from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])
print("✅ HuggingFace 로그인 완료")

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
# ## 과제 2. LoRA Configuration 설정
#
# 200 실습과 **완전히 동일한 LoRA 설정** 을 그대로 사용합니다. base 모델만 다른
# 동일 파이프라인이라는 것을 강조하는 의도된 설계입니다.
#
# | 파라미터 | 값 | 설명 |
# |---------|---|------|
# | `r` | 32 | Low-rank 행렬의 rank |
# | `lora_alpha` | 32 | Scaling factor |
# | `target_modules` | `"all-linear"` | 모든 linear layer 에 LoRA 적용 |
# | `lora_dropout` | 0.05 | 과적합 방지 |
# | `task_type` | `CAUSAL_LM` | 다음 토큰 예측 (생성 태스크) |

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
# ## 과제 3. Base 모델 (Gemma 3 1B) + 토크나이저 로드 + 메모리 절약 설정
#
# **google/gemma-3-1b-it**
# - Google Gemma 3 시리즈의 경량 instruction-tuned 모델 (10억 파라미터)
# - 다국어 코퍼스로 사전학습 (한국어 포함)
# - 1B 라 Colab T4 에서도 LoRA 학습이 무난하게 돌아감
# - HuggingFace: https://huggingface.co/google/gemma-3-1b-it
#
# > 참고: 본 실습은 200 (Qwen) 과의 직접 비교를 위해 chat template 없이 raw 프롬프트
# > 포맷 (`문맥:/질문:/답변:`) 을 그대로 사용합니다. LoRA 가 이 포맷에 맞춰 적응합니다.
#
# **할 일**:
# - `AutoTokenizer` 와 `AutoModelForCausalLM` 으로 모델/토크나이저를 로드하세요.
# - `pad_token` 이 없으면 `eos_token` 으로 채우세요.
# - **T4 OOM 방지** 를 위해 다음 세 줄을 추가하세요:
#   - `model.config.use_cache = False` (checkpointing 과 충돌 방지)
#   - `model.gradient_checkpointing_enable()`
#   - `model.enable_input_require_grads()` (LoRA 학습 시 입력 grad 활성화)
#
# **힌트**: Gemma 3 는 vocab 크기가 262K 로 매우 커서 logits 텐서 (batch × seq × vocab)
# 의 메모리가 큽니다. 200 (Qwen) 노트북보다 한 단계 더 강한 메모리 절약 옵션이 필요합니다.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-1b-it"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Base 모델 로드 (fp16 으로 메모리 절약)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Gradient checkpointing 준비 (T4 16GB 에서 OOM 방지)
# - Gemma 3 는 vocab 크기가 262K 로 커서 logits 메모리가 큼
# - activation 을 저장하지 않고 backward 때 재계산하여 메모리 절약
model.config.use_cache = False              # checkpointing 과 충돌 방지
model.gradient_checkpointing_enable()
model.enable_input_require_grads()          # LoRA 학습을 위해 입력 grad 활성화

print(f"모델: {model_name}")
print(f"파라미터 수: {model.num_parameters():,}")
print(f"토크나이저 vocab 크기: {len(tokenizer):,}")

# %% [markdown]
# **관찰 포인트**
# - `vocab 크기 ≈ 262K` 가 출력되는지 확인하세요 (Qwen 의 약 1.5~2배).
# - gradient checkpointing 은 "activation 을 저장 안 하고 backward 때 다시 계산"
#   하는 트릭이라, **메모리는 줄이는 대신 학습 시간이 약 30% 정도 늘어납니다**.
#   T4 16 GB 에서 1B 모델을 돌리기 위한 사실상 필수 옵션.

# %% [markdown]
# ---
# ## 과제 4. LoRA adapter 주입 (`get_peft_model`)
#
# 200 노트북과 완전히 동일한 한 줄 — `get_peft_model(model, lora_config)`.
# base 모델만 바뀌어도 이 단계는 그대로 통합니다.
#
# **할 일**:
# - `get_peft_model` 로 LoRA adapter 가 붙은 모델을 만드세요.
# - `print_trainable_parameters()` 로 학습 파라미터 비율을 확인하고, 200 의 Qwen
#   결과(보통 < 1%) 와 비교해 보세요.

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
# - 학습 가능한 파라미터의 절대 개수는 Gemma 가 Qwen 보다 더 큽니다 (base 가 2배 정도 크니까).
# - 그러나 **비율** 은 두 모델 모두 1% 미만으로 비슷합니다 — 이것이 LoRA 의 일관된 특성입니다.

# %% [markdown]
# ---
# ## 과제 5. KorQuAD 데이터 준비 (200 과 동일 + `MAX_LENGTH` 만 조정)
#
# 데이터셋, 샘플 크기, 프롬프트 포맷, 라벨 마스킹 — 모두 200 노트북과 동일합니다.
# **딱 한 가지** 만 다릅니다: `MAX_LENGTH` 를 256 → **384** 로 늘립니다.
#
# > Gemma 토크나이저는 한국어를 Qwen 대비 더 잘게 쪼개는 편이라, 같은 문장이라도
# > 토큰 수가 더 많이 나옵니다. `MAX_LENGTH=256` 으로 두면 잘림이 자주 발생합니다.
#
# **할 일**:
# - `load_dataset("KorQuAD/squad_kor_v1", split="train")` 로 데이터를 로드하고
#   5,000 + 200 개를 샘플링하세요.
# - `format_qa()` 로 한국어 프롬프트 포맷을 만들고 `tokenize_fn()` 으로 라벨 마스킹을 적용하세요.

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
# ### 데이터 포맷팅 + 토크나이징
#
# > Gemma 토크나이저는 한국어를 Qwen 대비 더 잘게 쪼개는 편이라 `MAX_LENGTH` 를
# > 256 → **384** 로 약간 늘립니다.

# %%
MAX_LENGTH = 384


def format_qa(example):
    """KorQuAD 예제를 생성형 QA 텍스트로 변환"""
    context = example["context"][:300]  # context 길이 제한
    question = example["question"]
    answer = example["answers"]["text"][0]

    # 한국어 프롬프트 포맷 (200 노트북과 동일)
    prompt = f"문맥: {context}\n질문: {question}\n답변:"
    full_text = f"{prompt} {answer}{tokenizer.eos_token}"

    return {"prompt": prompt, "full_text": full_text}


# 포맷 적용
train_dataset = train_dataset.map(format_qa)
eval_dataset = eval_dataset.map(format_qa)

print("=== 변환된 예시 ===")
print(train_dataset[0]["full_text"][:400])

# %% [markdown]
# PyTorch 의 CrossEntropyLoss 에서 `ignore_index` 의 기본값이 `-100` 입니다.
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
# ---
# ## 과제 6. 학습 전 (baseline) 성능 측정
#
# 학습을 시작하기 전에 정성(생성 결과) 과 정량(Perplexity) 양쪽으로 기준선을 잡습니다.
#
# **할 일**:
# - `generate_answer()` 함수를 정의하세요 (문맥 + 질문 → 답변).
# - 평가 데이터 5개에 대해 학습 전 예측을 `baseline_preds` 로 저장하세요.

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
# - Gemma 3 의 baseline 한국어 QA 성능은 Qwen 보다 더 낮을 수 있습니다 — 다국어
#   사전학습이지만 한국어 비중이 상대적으로 적은 편입니다. 그만큼 **학습 전/후 격차**
#   가 크게 드러나는 좋은 비교 대상이 됩니다.

# %% [markdown]
# ---
# ## 과제 7. Trainer 준비 + 학습 전 Perplexity 측정
#
# **할 일**:
# - `TrainingArguments` 를 설정하세요. 200 (Qwen) 과의 차이는 다음 세 가지입니다:
#   - `per_device_train_batch_size=2` (200 은 8)
#   - `gradient_accumulation_steps=8` (200 은 2) → effective batch = 16 으로 동일
#   - `gradient_checkpointing=True` + `gradient_checkpointing_kwargs={"use_reentrant": False}`
# - `Trainer` 를 만들고, **학습 시작 전에** `trainer.evaluate()` 로 baseline Perplexity 를 측정하세요.
#
# **힌트**: micro batch 를 작게 가져가는 대신 grad accumulation 으로 effective batch
# 를 같게 유지하는 것이 핵심입니다. 200 과 동일한 학습 신호량을 받으면서 메모리만 줄이는 트릭.
#
# > **Perplexity = 모델이 다음 토큰을 예측할 때 느끼는 "혼란도"**
# > Perplexity 가 100 이면 → 모델이 매 토큰마다 "100개 후보 중 하나" 를 찍는 수준의 불확실성.
# > - 1 : 다음 토큰을 100% 확신, 10 : 꽤 잘 알고 있음, 100 : 많이 헷갈림

# %%
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./lora-korquad-gemma-checkpoints",
    num_train_epochs=3,
    # T4 16GB 메모리에 맞춰 micro batch 는 작게, gradient accumulation 으로 보충
    # effective batch = 2 * 8 = 16 (200 노트북과 동일)
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    # Gradient checkpointing: activation 재계산으로 메모리 절약
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
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
# ## 과제 8. LoRA Fine-tuning 실행
#
# **할 일**: `trainer.train()` 으로 학습을 실행하고 step 수/loss/소요 시간을 출력하세요.
#
# **힌트**: Gemma 1B 는 Qwen 0.5B 보다 모델이 크고 gradient checkpointing 이 켜져 있어
# 같은 데이터·epoch 라도 200 보다 학습이 다소 오래 걸립니다. T4 에서 보통 20~40 분.

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
# ## 과제 9. 학습 전/후 비교 (Perplexity + Loss curve + 생성 결과)
#
# **할 일**:
# - 학습 후 Perplexity 를 다시 측정하고 baseline 과의 개선율(%) 을 계산하세요.
# - `trainer.state.log_history` 에서 training loss 를 뽑아 곡선으로 그리세요.
# - 학습 전/후 예측을 같은 5개 샘플에 대해 나란히 출력하고 정답 키워드 포함 여부를
#   ✅/❌ 로 표시하세요.

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
plt.title('LoRA Fine-tuning Loss Curve (Gemma 3)')
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
# **관찰 포인트 — 200 (Qwen) 노트북과의 비교**
# - 같은 LoRA 설정·같은 데이터·같은 effective batch 로 학습했는데, 두 모델의
#   학습 전/후 perplexity 절대값과 정답 포함 비율이 다르게 나옵니다.
# - **base 모델의 한국어 사전학습 품질**이 결과에 가장 큰 영향을 줍니다.
# - LoRA 가 만능 보정기는 아니라는 점도 함께 확인하세요 — base 가 약하면 같은 양의
#   adapter 학습으로는 격차를 다 메우지 못할 수도 있습니다.

# %% [markdown]
# ---
# ## 과제 10. 모델 저장 & 로드 & Merge
#
# **할 일**:
# - `lora_model.save_pretrained("./gemma3-lora-korquad")` 로 adapter 를 저장하세요.
# - 저장된 파일 목록과 크기를 출력해 base model (~2 GB) 대비 얼마나 작은지 확인하세요.
# - `PeftModel.from_pretrained` 로 adapter 를 다시 로드해 동일 추론이 되는지 확인하세요.
# - `merge_and_unload()` 로 adapter 를 base 에 합쳐 단일 모델로 만들어 보세요.
#
# > ⚠️ `merge_and_unload` 후에는 adapter 를 다시 분리할 수 없습니다.

# %%
# adapter 저장
save_path = "./gemma3-lora-korquad"
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
print(f"(참고: base model 전체는 ~2 GB)")

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
# ---
# ## 종합 정리
#
# | 단계 | 200 (Qwen2.5-0.5B) | **210 (Gemma 3 1B)** | 차이의 이유 |
# |------|---------------------|-----------------------|-------------|
# | 인증 | 불필요 | `HF_TOKEN` 로그인 필수 | gated 모델 |
# | 메모리 옵션 | fp16 + auto | **fp16 + gradient checkpointing** | vocab 262K → logits 텐서 큼 |
# | batch | micro 8 × accum 2 = 16 | **micro 2 × accum 8 = 16** | effective batch 동일, 메모리만 절약 |
# | `MAX_LENGTH` | 256 | **384** | Gemma 한국어 토큰이 잘게 쪼개짐 |
# | LoRA 설정 | r=32, all-linear, dropout 0.05 | **완전히 동일** | LoRA 는 base-agnostic |
# | 데이터 / 라벨 마스킹 | KorQuAD 5000, 프롬프트 -100 | **완전히 동일** | 파이프라인 재사용 가능 |
#
# **핵심 메시지**:
# - LoRA 의 큰 매력 중 하나는 **base 모델에 거의 무관하게 같은 파이프라인이 통한다** 는 점입니다.
#   `get_peft_model`, 학습 코드, 평가 코드가 그대로 재사용됩니다.
# - 대신 base 모델의 특성 (vocab 크기, 토큰화 효율, 한국어 사전학습 비중) 에 따라
#   **메모리 옵션과 `MAX_LENGTH`** 같은 환경 설정은 조정해야 합니다.
# - 학습 전/후 결과를 200 과 함께 표로 비교해 보면, "base 가 강한 한국어 모델 +
#   가벼운 LoRA" 가 가장 비용 효율적이라는 직관이 자연스럽게 형성됩니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. 200 의 Qwen 학습 결과와 본 노트북의 Gemma 학습 결과에서 **같은 5개 샘플** 의
#    예측을 한 표로 모아 비교하세요. 어느 쪽이 어떤 유형의 질문에 더 강한지 분석해 보세요.
# 2. `MAX_LENGTH` 를 256 으로 줄여 보고 평균 truncation 비율과 perplexity 변화를 확인하세요.
# 3. `gradient_checkpointing=False` 로 끄고 micro batch 를 줄여 OOM 이 발생하는 지점을 직접 관찰하세요.
# 4. base model 을 `google/gemma-3-4b-it` 로 바꾸려면 어떤 추가 옵션 (예: 4-bit 양자화)
#    이 필요한지 조사하고, 가능한 범위에서 학습을 시도해 보세요.
#
# ### 참고 링크
# - [KorQuAD 공식](https://korquad.github.io/)
# - [Gemma 3 모델 카드](https://huggingface.co/google/gemma-3-1b-it)
# - [PEFT 공식 문서](https://huggingface.co/docs/peft/en/index)
# - [LoRA 개발자 가이드](https://huggingface.co/docs/peft/en/developer_guides/lora)

# %%
