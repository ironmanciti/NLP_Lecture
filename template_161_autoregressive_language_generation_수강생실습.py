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
# # 160. 수강생 실습 — Autoregressive 문장 생성 (HyperCLOVAX)
#
# ## 학습 목표
# 네이버 HyperCLOVAX 모델로 한국어 문장 생성을 직접 실습하며,
# **GPT 계열 LLM 의 자동회귀(autoregressive) 생성 원리**를 이해합니다.
#
# 이 실습에서는 같은 작업(`"옛날 옛적에"` 이어쓰기) 을 두 가지 방식으로 합니다.
#
# | 방식 | 한 줄 요약 | 특징 |
# |------|-----------|------|
# | **① `model.generate(...)`** | 한 번 호출로 끝 | 내부적으로 반복하지만 사용자는 1줄 |
# | **② 직접 `while` 루프** | 토큰 하나씩 예측 → 이어 붙이기 | autoregressive 원리를 눈으로 확인 |
#
# 두 결과를 비교하며 "고수준 API 는 결국 무엇을 자동으로 해주는가"를 체감하세요.
#
# ## Autoregressive 란?
# - 이전에 생성된 토큰들을 입력으로 받아 **다음 토큰의 확률 분포** 를 예측.
# - 그 분포에서 토큰을 하나 골라(보통 argmax 또는 sampling) 시퀀스에 이어 붙임.
# - 이어 붙인 더 긴 시퀀스를 다시 입력으로 넣고 그 다음 토큰을 예측 — 반복.
# - 이 단순한 루프가 GPT, HyperCLOVAX 등 모든 디코더형 LLM 의 핵심 메커니즘입니다.
#
# > **실행 환경**: Colab GPU 권장. 0.5B 모델이라 CPU 로도 동작은 하지만 매우 느립니다.

# %% [markdown]
# ---
# ## 0. 라이브러리 설치
# Colab 에서 실행한다면 아래 설치 명령을 먼저 실행하세요.

# %%

# %% [markdown]
# ---
# ## 과제 1. 모델과 토크나이저 로드
#
# 네이버 HyperCLOVAX SEED 시리즈 중 가장 작은 0.5B 모델을 사용합니다.
# Instruct 버전이라 시스템/유저/어시스턴트 역할이 구분된 chat 형식 입력을 받습니다.
#
# **할 일**:
# - `AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")` 로 모델을 로드하세요.
# - 같은 이름으로 `AutoTokenizer` 도 로드하세요.
#
# **힌트**:
# - `AutoModelForCausalLM` = Causal Language Modeling (다음 단어 예측) 용 모델 클래스.
# - `device_map="auto"` 가 GPU 가 있으면 자동으로 GPU 에 모델을 올려 줍니다.

# %%
# 사용할 사전 학습 모델 이름 (네이버 HyperCLOVA X SEED 시리즈 중 하나)
# 사전 학습된 언어 모델 로드
# AutoModelForCausalLM: "Causal Language Modeling" (다음 단어 예측) 용 모델 클래스
# device_map="auto" → GPU 가 있으면 자동으로 GPU 에 올려줌 (없으면 CPU 사용)
# 해당 모델과 호환되는 토크나이저 로드
# 토크나이저: 문장을 토큰 단위로 분해해 숫자 ID 로 변환하거나 다시 문자열로 복원

# %% [markdown]
# **관찰 포인트**
# - 처음 호출 시 모델 가중치가 HuggingFace Hub 에서 다운로드되며, 두 번째부터는 캐시 사용.
# - `device_map="auto"` 덕분에 GPU 환경/메모리에 맞춰 자동으로 배치됩니다.

# %% [markdown]
# ---
# ## 과제 2. Chat 템플릿으로 입력 만들기 + 토큰 구조 확인
#
# HyperCLOVAX 같은 Instruct 모델은 단순 문장이 아니라 **역할이 표시된 대화 형식**을
# 입력으로 받습니다. `tokenizer.apply_chat_template(...)` 가 이 변환을 자동으로 해 줍니다.
#
# **할 일**:
# - `chat = [{"role": "user", "content": "옛날 옛적에"}]` 를 만드세요.
# - `apply_chat_template(chat, add_generation_prompt=True, return_dict=True, return_tensors="pt")`
#   로 모델 입력을 만드세요.
# - 입력을 `model.device` 로 옮기고, `input_ids` 와 토큰 개수, 디코딩 결과를 출력해 확인하세요.
#
# **힌트**:
# - `add_generation_prompt=True` → 모델이 이어서 응답을 생성하도록 마지막에
#   "<|assistant|>" 같은 프롬프트를 자동으로 붙여 줍니다.
# - 같은 문장 `"옛날 옛적에"` 만 단독 토큰화해서 비교해 보면, chat 템플릿이 어떤
#   특수 토큰을 앞뒤로 붙이는지 한눈에 보입니다.

# %%
# 대화 시작 부분 설정
# chat 형식으로 대화 내용 구성
# 'role' 은 대화 참여자 구분 (system / user / assistant)
# chat 템플릿을 모델 입력 형식으로 변환
# add_generation_prompt=True → 모델이 이후 이어서 문장을 생성하도록 프롬프트 추가
# return_dict=True → dictionary 형태로 반환
# return_tensors="pt" → PyTorch 텐서 형태로 반환
# 모델과 동일한 디바이스(GPU/CPU) 에 입력 데이터 로드
# 전체 입력 토큰 확인
# 전체 입력 토큰을 다시 디코딩하여 원문 형태로 복원
# -------------------------------------------------------------
# 사용자 입력 부분만 별도로 토큰화 및 디코딩 (비교용)
# -------------------------------------------------------------
# 단순히 user_content(문장)만 토큰화하여 확인
# 토큰화된 결과와 토큰 개수 출력
# 다시 토큰을 텍스트로 복원

# %% [markdown]
# **관찰 포인트**
# - chat 템플릿이 적용된 시퀀스는 `<|user|>`, `<|assistant|>` 같은 **역할 표시 특수 토큰**이
#   원본 문장 앞뒤로 붙어 있습니다.
# - 단순 `encode("옛날 옛적에")` 의 토큰 개수보다 훨씬 깁니다 — 이 차이가 곧
#   "Instruct 모델이 학습한 대화 포맷" 의 오버헤드입니다.

# %% [markdown]
# ---
# ## 과제 3. `model.generate(...)` 로 한 번에 문장 생성
#
# HuggingFace 의 고수준 API 인 `model.generate(...)` 는 내부적으로 autoregressive
# 루프를 돌려 시퀀스를 만들어 줍니다. 사용자가 직접 루프를 작성할 필요가 없습니다.
#
# **할 일**:
# - `model.generate(**inputs, max_length=100, ...)` 로 문장을 생성하세요.
# - `repetition_penalty`, `eos_token_id`, `pad_token_id` 옵션의 의미를 주석으로 확인하세요.
# - 결과 토큰을 `batch_decode(..., skip_special_tokens=True)` 로 자연어로 복원하세요.
# - HyperCLOVAX 특유의 종료 표시(`<|endofturn|>`, `<|stop|>`)는 잘라내세요.
#
# **힌트**:
# - `repetition_penalty=1.2` — 같은 단어 반복을 억제. 1.0 이면 패널티 없음.
# - `eos_token_id` 를 만나면 생성이 조기 종료됩니다.

# %%
# 문장 생성 (모델이 이어서 텍스트를 생성하도록 함)
# 생성된 출력 토큰 ID 확인

# %%
# 생성된 문장을 텍스트로 디코딩 (HyperCLOVA-X 방식)
# batch_decode(): 여러 문장을 한 번에 디코딩할 수 있음
# skip_special_tokens=True → <bos>, <eos> 등의 특수 토큰은 제거
# 생성된 결과에 불필요한 종료 토큰(<|endofturn|>, <|stop|>)이 포함되어 있으면 잘라냄
# HyperCLOVA-X 계열 모델은 대화 종료를 나타내는 특수 토큰을 출력할 수 있음
# 최종 생성 결과 출력

# %% [markdown]
# **관찰 포인트**
# - `model.generate` 가 내부적으로 한 토큰씩 예측하며 시퀀스를 늘려갔지만,
#   사용자 입장에서는 함수 한 번 호출로 끝났습니다.
# - 같은 입력으로 여러 번 실행해 보세요. `do_sample=False` (기본) 일 때는 결과가
#   항상 똑같습니다 (argmax greedy decoding). 확률적 샘플링을 원하면
#   `do_sample=True, temperature=0.8` 등을 추가합니다.

# %% [markdown]
# HyperCLOVAX 는 자체적으로 autoregressive 모델입니다.
# 위의 `model.generate` 메서드는 이미 autoregressive 방식으로 문장을 생성합니다.
# 그러나 이를 **명시적으로 보여주기 위해**, 다음 두 과제에서 직접 토큰을 하나씩
# 생성하는 autoregressive 코드를 작성해 봅니다.

# %% [markdown]
# ---
# ## 과제 4. 모델 forward pass 로 logits 직접 확인
#
# `model.generate` 가 내부에서 호출하는 핵심 연산은 `model(**inputs)` (forward pass) 입니다.
# 이 호출이 반환하는 **logits** 가 "다음 토큰이 무엇이 될지에 대한 점수" 입니다.
#
# **할 일**:
# - 과제 2와 동일하게 chat 템플릿으로 입력을 만드세요.
# - `predictions = model(**inputs)` 를 호출해 `predictions.logits` 의 shape 을 출력하세요.
# - shape 의 각 차원이 무엇을 의미하는지 주석으로 정리하세요.
#
# **힌트**:
# - `logits.shape == (batch_size, sequence_length, vocab_size)`
# - 마지막 위치의 logits (`logits[0, -1]`) 에서 argmax 를 취하면 "다음 토큰" 이 됩니다.

# %%
# 사용자 입력 문장 정의
# 대화(chat) 형식의 입력 구성
# - role: 대화 참여자의 역할 ("system", "user", "assistant" 중 하나)
# - content: 각 발화의 실제 텍스트
# chat 데이터를 모델 입력 형식으로 변환
# apply_chat_template() 함수는 모델이 학습한 대화 템플릿(<|user|>, <|assistant|> 등)을 자동으로 추가
# add_generation_prompt=True → 모델이 이어서 대답을 생성할 수 있도록 마지막에 assistant 프롬프트를 추가
# return_dict=True → 반환 값을 dict 형태로 (예: {'input_ids': ..., 'attention_mask': ...})
# return_tensors="pt" → PyTorch 텐서 형태로 반환 (모델 입력용)
# 모델이 사용 중인 디바이스(GPU 또는 CPU) 에 입력 텐서를 로드
# 토크나이저 처리 결과(입력 텐서 구조) 출력
# 'input_ids' : 모델이 처리할 토큰 ID 시퀀스
# 'attention_mask' : 실제 토큰과 패딩 구분용 마스크

# %%
# 모델 추론(Forward Pass)
# 입력 데이터를 모델에 전달하여 예측값(logits) 을 계산
# **inputs → 딕셔너리를 언패킹하여 전달 (input_ids, attention_mask 등 포함)
# 모델의 출력 중 logits(로짓값) 추출
# logits 은 각 토큰 위치마다 다음 단어(토큰) 가 될 확률의 원시 점수(raw score)
# 출력 텐서의 차원(shape) 확인
# (batch_size, sequence_length, vocab_size)
# - batch_size: 입력 문장 수
# - sequence_length: 입력 토큰 길이
# - vocab_size: 모델의 어휘 집합 크기 (예: 50,000개 등)

# %% [markdown]
# **관찰 포인트**
# - `logits.shape[2]` (vocab_size) 가 매우 큽니다 (수만~수십만). 모델은 매 위치마다
#   "vocab 전체에 대한 점수" 를 출력합니다.
# - 다음 토큰을 만들 때 우리가 실제로 사용하는 것은 **`logits[0, -1, :]`** — 즉
#   batch 0번 샘플의 **마지막 위치** 의 점수 한 줄뿐입니다.

# %% [markdown]
# ---
# ## 과제 5. Autoregressive 한 토큰씩 직접 생성
#
# 이제 `model.generate` 가 내부에서 하는 일을 **직접 손으로** 구현합니다.
#
# **할 일**:
# - `input_ids_concat` 에 현재까지의 시퀀스를 담고 `while` 루프로 토큰을 늘려가세요.
# - 매 루프마다 (1) forward pass → (2) `logits[0, -1]` 의 argmax 로 다음 토큰 선택 →
#   (3) 기존 시퀀스에 cat 으로 이어 붙이기 를 수행하세요.
# - `max_length` 에 도달하면 종료합니다.
# - 마지막으로 디코딩해서 자연어로 출력하고, 특수 토큰 (`<|endofturn|>`, `<|stop|>`) 은 잘라내세요.
#
# **힌트**:
# - `predicted_token = torch.argmax(logits[0, -1]).item()`
# - `torch.cat([seq, torch.tensor([[predicted_token]], device=seq.device)], dim=1)` 로 이어 붙임.
# - 매 스텝마다 `attention_mask` 도 길이를 맞춰 새로 만들어 줘야 합니다 (`torch.ones_like(input_ids_concat)`).

# %%
# Autoregressive(자가회귀적) 방식으로 문장 생성
# → 모델이 한 번에 한 토큰씩 다음 단어를 예측하면서 문장을 점진적으로 완성하는 방식
# 입력 길이가 최대 길이에 도달할 때까지 반복
    # 현재까지의 입력 토큰을 모델 입력 형식으로 준비
    # attention_mask 가 있을 경우, 전체 길이에 맞게 1로 채워서 추가
    # (1은 실제 토큰, 0은 패딩을 의미함)
    # 모델 추론(Forward Pass)
    # 현재까지의 토큰을 입력으로 주고 다음 토큰의 확률분포(logits) 계산
    # 가장 마지막 토큰 위치의 logits 에서 확률이 가장 높은 토큰 선택
    # torch.argmax(logits[0, -1]) → 마지막 시퀀스의 마지막 토큰에 대한 예측 결과
    # print(predicted_token)  # 디버깅용: 예측된 토큰 ID 확인
    # 생성된 토큰을 기존 입력 시퀀스 뒤에 이어붙이기
    # 현재까지 생성된 전체 토큰 시퀀스 출력 (디버깅용)

# %%
# 생성된 문장을 텍스트로 디코딩
# input_ids_concat[0] : 모델이 생성한 전체 토큰 시퀀스 (1차원 텐서)
# skip_special_tokens=True → <bos>, <eos> 등 특수 토큰은 제거하고 자연어만 복원
# HyperCLOVA-X 계열 모델은 대화 종료나 중단 지점을 나타내는 특수 토큰을 출력할 수 있음
# 예: "<|endofturn|>", "<|stop|>"
# 이런 토큰이 포함되어 있다면 해당 지점까지만 문장을 남기고 이후는 제거
# 최종 생성된 문장 출력

# %% [markdown]
# **관찰 포인트 — 두 방식 비교**
# - 과제 3 의 `model.generate` 결과와 과제 5 의 직접 루프 결과가 (greedy decoding 인 한)
#   거의 같은 문장이 되어야 합니다. 차이가 있다면 `repetition_penalty` 적용 여부 때문입니다.
# - 매 스텝마다 시퀀스 전체를 다시 forward 시키는 이 직접 루프는 **매우 비효율적**입니다.
#   실제 `model.generate` 는 **KV cache** 를 활용해 이전 계산을 재사용하므로 훨씬 빠릅니다.
# - 그래도 이 직접 구현 덕분에 "LLM 이 결국 argmax 를 반복할 뿐" 이라는 핵심 원리가
#   체감됩니다 — 모델의 신비감을 한 꺼풀 벗기는 단계입니다.

# %% [markdown]
# ---
# ## 종합 정리
#
# | 단계 | 핵심 도구 | 자동회귀 생성의 핵심 포인트 |
# |------|-----------|------------------------------|
# | ① 모델 로드 | `AutoModelForCausalLM` + `device_map="auto"` | Causal LM = 다음 토큰 예측 |
# | ② 입력 만들기 | `tokenizer.apply_chat_template(...)` | Instruct 모델은 역할 토큰을 함께 입력 |
# | ③ 고수준 생성 | `model.generate(...)` | 사용자는 1줄, 내부에서 루프 처리 |
# | ④ logits 확인 | `model(**inputs).logits` shape | `(batch, seq, vocab)` — 다음 토큰의 후보 점수 |
# | ⑤ 직접 루프 | `argmax(logits[0,-1])` → cat → repeat | autoregressive 생성을 손으로 재현 |
#
# **핵심 메시지**:
# - 거대 언어 모델의 문장 생성은 결국 **"마지막 위치의 logits 에서 다음 토큰을 골라
#   이어 붙이는 루프"** 일 뿐입니다.
# - `model.generate(...)` 는 이 루프에 KV cache, sampling, beam search, repetition
#   penalty, EOS 종료 등 다양한 최적화를 더한 고수준 함수입니다.
# - chat 템플릿은 모델이 학습된 포맷에 맞춰 입력을 자동 변환해 주는 도구로, Instruct
#   모델에서는 사용 여부에 따라 출력 품질이 크게 달라집니다.

# %% [markdown]
# ---
# ## 추가 실습 (선택 과제)
#
# 1. `user_content` 를 다른 문장으로 바꿔(예: `"한국의 수도는"`, `"인공지능이란"`) 두 방식의
#    생성 결과를 비교하세요.
# 2. `model.generate(..., do_sample=True, temperature=0.8, top_p=0.9)` 로 옵션을 바꿔
#    같은 입력으로 여러 번 호출했을 때 결과가 어떻게 달라지는지 확인하세요.
# 3. 과제 5 의 직접 루프에서 `argmax` 대신 `torch.multinomial(F.softmax(logits[0,-1], dim=-1), 1)`
#    로 sampling 방식을 바꿔 보고, 같은 입력에 대해 결과가 매번 달라지는지 관찰하세요.
# 4. 과제 5 의 루프에 "예측 토큰이 `eos_token_id` 면 즉시 종료" 조건을 추가해
#    `max_length` 이전에도 자연스럽게 문장이 끝나도록 개선하세요.

# %%
