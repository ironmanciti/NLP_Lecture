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
# # PEFT LoRA 튜토리얼: 한국어 QA Fine-tuning 실습 (Gemma 3)
#
# **데이터**: KorQuAD 1.0 (한국어 SQuAD)
# **모델**: google/gemma-3-1b-it (Google Gemma 3, 10억 파라미터)
#
# 이 노트북은 [200_PEFT_LoRA_KorQuAD_Finetuning.ipynb](200_PEFT_LoRA_KorQuAD_Finetuning.ipynb) 의
# Qwen2.5-0.5B를 **Gemma 3 (1B)** 로 교체한 버전입니다.
# 동일한 LoRA 파이프라인이 다른 base 모델에서도 그대로 동작함을 확인할 수 있습니다.
#
# 이 노트북에서는 다음을 실습합니다:
# 1. Gemma 3 모델에 LoRA adapter 주입
# 2. KorQuAD 데이터로 QA fine-tuning 수행
# 3. 학습 전/후 한국어 QA 성능 비교 (정성 + 정량 평가)
# 4. 모델 저장 및 로드
#
# > ⚙️ Colab에서 **GPU 런타임**을 선택하세요: 런타임 → 런타임 유형 변경 → T4 GPU
# > 🔑 Gemma는 **gated 모델**입니다. 먼저 https://huggingface.co/google/gemma-3-1b-it 에서
# > 라이센스에 동의하고, HuggingFace에서 발급받은 토큰을 Colab의 **Secrets (🔑 자물쇠 아이콘)** 에
# > `HF_TOKEN` 으로 등록해 주세요.
# > 💾 Gemma 3는 vocab 크기가 262K로 커서 학습 시 logits 메모리가 큽니다.
# > T4 (16GB) 에서 OOM을 피하기 위해 micro batch=2 + gradient_accumulation=8 (effective=16)
# > + gradient_checkpointing을 사용합니다.

# %% [markdown]
# ## 0. 환경 설정

# %%
# 필요한 라이브러리 설치
# Gemma 3는 transformers 4.50+ 가 필요하므로 -U 로 업그레이드합니다.
# !pip install -q -U transformers peft accelerate datasets
# !pip install -q --upgrade torchao

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
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=32,                                # rank: low-rank 행렬의 차원
    target_modules="all-linear",          # 모든 linear layer에 LoRA 적용
    task_type=TaskType.CAUSAL_LM,        # Causal Language Modeling
    lora_alpha=32,                       # scaling factor
    lora_dropout=0.05                    # dropout
)

print(lora_config)

# %% [markdown]
# ## 2. Base Model 로드
#
# **google/gemma-3-1b-it**
# - Google Gemma 3 시리즈의 경량 instruction-tuned 모델 (10억 파라미터)
# - 다국어 코퍼스로 사전학습 (한국어 포함)
# - 1B라 Colab T4에서 LoRA 학습이 무난하게 돌아감
# - HuggingFace: https://huggingface.co/google/gemma-3-1b-it
#
# > 참고: 본 노트북은 200 과의 비교를 위해 chat template 없이 raw 프롬프트 포맷
# > (`문맥:/질문:/답변:`)을 그대로 사용합니다. LoRA가 이 포맷에 맞춰 적응합니다.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "google/gemma-3-1b-it"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Base 모델 로드 (fp16으로 메모리 절약)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 패딩 토큰 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

# Gradient checkpointing 준비 (T4 16GB에서 OOM 방지)
# - Gemma 3는 vocab 크기가 262K로 커서 logits 메모리가 큼
# - activation을 저장하지 않고 backward 때 재계산하여 메모리 절약
model.config.use_cache = False              # checkpointing과 충돌 방지
model.gradient_checkpointing_enable()
model.enable_input_require_grads()          # LoRA 학습을 위해 입력 grad 활성화

print(f"모델: {model_name}")
print(f"파라미터 수: {model.num_parameters():,}")
print(f"토크나이저 vocab 크기: {len(tokenizer):,}")

# %% [markdown]
# ## 3. PeftModel 생성
#
# `get_peft_model()`로 base model에 LoRA adapter를 주입합니다.
# 원래 weights는 freeze, LoRA A/B 행렬만 학습됩니다.

# %%
import torch
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()

# %%
# LoRA가 적용된 레이어 확인
trainable_params = [(name, param.shape) for name, param in lora_model.named_parameters() if param.requires_grad]
print(f"학습 가능한 파라미터 그룹 수: {len(trainable_params)}\n")
for name, shape in trainable_params[:10]:  # 처음 10개만
    print(f"  ✅ {name} | {shape}")
if len(trainable_params) > 10:
    print(f"  ... 외 {len(trainable_params) - 10}개")

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
# Colab T4에서 실습 가능한 크기로 샘플링
TRAIN_SIZE = 5000
EVAL_SIZE = 200

small_dataset = dataset.shuffle(seed=42).select(range(TRAIN_SIZE + EVAL_SIZE))
train_dataset = small_dataset.select(range(TRAIN_SIZE))
eval_dataset = small_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + EVAL_SIZE))

print(f"학습 데이터: {len(train_dataset)}개")
print(f"평가 데이터: {len(eval_dataset)}개")

# %% [markdown]
# ### 데이터 전처리
#
# QA를 Causal LM 포맷으로 변환합니다.
# 모델이 `답변:` 이후 부분만 생성하도록 **labels masking**을 적용합니다.
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
# PyTorch의 CrossEntropyLoss에서 ignore_index의 기본값이 -100입니다.
# label이 -100인 위치는 loss 계산에서 완전히 제외됩니다. 즉, 그 토큰은 "맞추든 틀리든 상관없다"는 의미입니다.

# %%
def tokenize_fn(examples):
    """토크나이징 + labels masking (프롬프트 부분은 -100으로)"""
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

print(f"토크나이징 완료")
print(f"  학습 데이터 컬럼: {tokenized_train.column_names}")
print(f"  입력 시퀀스 길이: {len(tokenized_train[0]['input_ids'])}")


# %% [markdown]
# ## 5. 학습 전 성능 확인 (baseline)
#
# 학습 전에 모델이 한국어 QA를 어떻게 하는지 확인합니다.
# **정성 평가**(생성 텍스트 비교)와 **정량 평가**(Perplexity)를 모두 수행합니다.

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
            pad_token_id=tokenizer.eos_token_id   # ← 이거 추가
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
# ## 6. Trainer 준비 & 학습 전 Perplexity 측정
#
# 학습 전에 eval perplexity를 먼저 측정해 둡니다.
# 학습 후와 비교하면 LoRA의 효과를 **수치로** 확인할 수 있습니다.

# %%
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

training_args = TrainingArguments(
    output_dir="./lora-korquad-gemma-checkpoints",
    num_train_epochs=3,
    # T4 16GB 메모리에 맞춰 micro batch는 작게, gradient accumulation 으로 보충
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

# %% [markdown]
# **Perplexity = 모델이 다음 토큰을 예측할 때 느끼는 "혼란도"**
#
# Perplexity가 100이면 → 모델이 매 토큰마다 "100개 후보 중 하나"를 찍는 수준의 불확실성을 느낀다는 뜻
# - 1 : 다음 토큰을 100% 확신, 10 : 꽤 잘 알고 있음, 100 : 많이 헷갈림

# %%
# ── 학습 전 Perplexity 측정 ──
baseline_eval = trainer.evaluate()
baseline_ppl = torch.exp(torch.tensor(baseline_eval['eval_loss'])).item()
print(f"📊 학습 전 Eval Loss: {baseline_eval['eval_loss']:.4f}")
print(f"📊 학습 전 Perplexity: {baseline_ppl:.2f}")

# %% [markdown]
# ## 7. LoRA Fine-tuning 실행

# %%
print("학습을 시작합니다...\n")
train_result = trainer.train()

print(f"\n{'=' * 40}")
print(f"✅ 학습 완료!")
print(f"  Total steps: {train_result.global_step}")
print(f"  Training loss: {train_result.training_loss:.4f}")
print(f"  학습 시간: {train_result.metrics['train_runtime']:.1f}초")

# %% [markdown]
# ## 8. 학습 전/후 Perplexity 비교
#
# Perplexity가 낮을수록 모델이 정답 토큰을 잘 예측한다는 의미입니다.

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
# ### 8-2. Training Loss Curve

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
# ### 8-3. 학습 전/후 생성 결과 비교
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
# ## 9. 모델 저장 & 로드

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
# ## 10. Merge & Unload (선택)
#
# 서빙 시 adapter를 base에 합쳐서 추론 속도를 개선할 수 있습니다.
#
# > ⚠️ merge 후에는 adapter를 분리할 수 없습니다.

# %%
merged_model = loaded_model.merge_and_unload()
print(f"Merged 모델 타입: {type(merged_model).__name__}")

pred = generate_answer(merged_model, sample["question"], sample["context"][:300])
print(f"\n질문: {sample['question']}")
print(f"예측: {pred}")
print(f"\n✅ Merged 모델 추론 성공")

# %% [markdown]
# ## 정리
#
# | 항목 | 값 |
# |------|---|
# | Base model | google/gemma-3-1b-it (10억 파라미터) |
# | 데이터셋 | KorQuAD 1.0 (한국어 QA, 5,000개 사용) |
# | LoRA rank | 32 |
# | Target modules | all-linear |
# | 학습 에폭 | 3 |
# | Adapter 크기 | 수십 MB (base model ~2GB 대비 극소) |
#
# ### 200 노트북과의 비교 포인트
# - **공통**: LoRA 파이프라인, 데이터 전처리, 평가 방식 모두 동일
# - **차이**:
#   - Base 모델: Qwen2.5-0.5B → Gemma 3 1B
#   - HF 토큰 인증 필요 (Gemma는 gated 모델)
#   - 토큰화 효율 차이로 `MAX_LENGTH` 를 256 → 384 로 상향
#   - 한국어 baseline 성능이 Qwen 대비 낮을 수 있어, 학습 전/후 격차가 더 크게 보일 수 있음
#
# ### 더 나은 결과를 위한 팁
# - `TRAIN_SIZE`를 늘리기 (전체 60K 사용 시 성능 대폭 향상)
# - `num_train_epochs`를 5~10으로 늘리기
# - `r=64`로 rank 올리기 (메모리 허용 시)
# - 더 큰 base model 사용 (예: gemma-3-4b-it — 4-bit 양자화 권장)
#
# ### 참고 링크
# - [KorQuAD 공식](https://korquad.github.io/)
# - [Gemma 3 모델 카드](https://huggingface.co/google/gemma-3-1b-it)
# - [PEFT 공식 문서](https://huggingface.co/docs/peft/en/index)
# - [LoRA 개발자 가이드](https://huggingface.co/docs/peft/en/developer_guides/lora)
