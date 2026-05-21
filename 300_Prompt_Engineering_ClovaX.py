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
# # 실습 — 5가지 prompt 기법 비교 (HyperCLOVAX 버전)
#
# **공통 태스크: 영화 리뷰의 감성 분류 + 그렇게 판단한 근거 키워드 추출**
#
# 동일한 영화 리뷰 데이터에 5가지 prompt 기법을 차례로 적용하고, 기법이 바뀔 때
# 정확도와 출력 형태가 어떻게 달라지는지 비교합니다. 모델은 Hugging Face 에서
# 내려받는 `HyperCLOVAX-SEED-Text-Instruct-1.5B` 를 사용합니다.
# (600 노트북은 동일한 실습을 Gemini API 로 수행합니다.)
#
# | 기법 | Prompt 핵심 | 관찰 포인트 |
# |------|-------------|-------------|
# | Zero-shot | 이 리뷰의 감성을 분류해줘 | 기본 정확도 측정 (baseline) |
# | Few-shot | 예시 3개를 보여준 뒤 동일 형식으로 답하게 함 | 예시가 정확도·일관성을 얼마나 올리는가 |
# | CoT | 단계적으로 판단 근거를 설명한 후 결론 | 추론 노출이 오류를 줄이는가 |
# | Self-Consistency | 동일 CoT prompt 를 5회 호출 후 다수결 | 단발성 오류 제거 효과 |
# | Structured Output | Pydantic 스키마로 sentiment / confidence / keywords 반환 | 후처리 코드 없이 바로 사용 가능한가 |
#
# > Colab 에서 **GPU 런타임**으로 실행하세요. (런타임 → 런타임 유형 변경 → T4 GPU)

# %% [markdown]
# ## 0. 환경 설정
#
# - Hugging Face 토큰 생성: https://huggingface.co/settings/tokens 에서 Read 권한 토큰 생성
# - 모델 접근 권한 요청: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B
#   페이지에서 "Access repository" 버튼 클릭
#
# ```
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
# ```

# %%
import os

from dotenv import load_dotenv
from huggingface_hub import login

# .env 파일의 HF_TOKEN 을 환경변수로 불러온다 (토큰을 코드에 직접 노출하지 않기 위함)
#load_dotenv()
#login(token=os.environ["HF_TOKEN"])

login(token="")

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# %% [markdown]
# ## 1. 공통 데이터 — 영화 리뷰 + 정답 라벨
#
# 5개 리뷰 중 마지막 문장은 **반어법**(칭찬하는 듯하지만 실제로는 혹평)이라,
# 기법별 성능 차이를 드러내는 함정 역할을 합니다.

# %%
reviews = [
    {"text": "배우들 연기는 좋았지만 스토리가 너무 뻔하고 지루했다.", "label": "부정"},
    {"text": "시간 가는 줄 모르고 봤다. 올해 본 영화 중 단연 최고!", "label": "긍정"},
    {"text": "특수효과는 화려한데 딱 거기까지다. 내용이 텅 비어 있다.", "label": "부정"},
    {"text": "기대 없이 봤는데 의외로 깊은 감동이 있었고 여운이 길었다.", "label": "긍정"},
    {"text": "와 정말 대단한 영화네요, 두 시간 내내 졸기만 했습니다.", "label": "부정"},
]

for i, r in enumerate(reviews):
    print(f"{i + 1}. ({r['label']}) {r['text']}")

# %% [markdown]
# ## 2. 공통 헬퍼 함수
#
# 모든 기법이 같은 모델 호출 함수를 사용합니다. Self-Consistency 에서 매 호출마다
# 다른 결과가 나오도록 `do_sample=True`(확률적 샘플링) 옵션도 지원합니다.

# %%
def generate_response(system_content, user_content, max_new_tokens=512,
                      repetition_penalty=1.2, do_sample=False, temperature=0.8):
    """HyperCLOVAX 모델로 응답을 생성한다.

    do_sample=False -> 항상 같은 결과 (그리디 디코딩)
    do_sample=True  -> 호출마다 다른 결과 (샘플링) — Self-Consistency 에 사용
    """
    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]
    inputs = tokenizer.apply_chat_template(
        chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    inputs = inputs.to(model.device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    if do_sample:
        # 샘플링 활성화 — 같은 prompt 라도 호출마다 다른 추론 경로가 나온다
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)

    output_ids = model.generate(**inputs, **gen_kwargs)

    # 입력 프롬프트를 제외하고 새로 생성된 토큰만 디코딩
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    output_text = tokenizer.decode(generated, skip_special_tokens=True)

    # HyperCLOVAX 종료 토큰이 남아 있으면 잘라낸다
    for stop_str in ["<|endofturn|>", "<|stop|>"]:
        if stop_str in output_text:
            output_text = output_text.split(stop_str)[0]
    return output_text.strip()


# %%
def parse_label(text):
    """모델 출력 문자열에서 '긍정' / '부정' 라벨을 추출한다.

    CoT 처럼 결론이 맨 뒤에 오는 경우를 위해, 가장 마지막에 등장한 단어를 채택한다.
    """
    pos = text.rfind("긍정")
    neg = text.rfind("부정")
    if pos == -1 and neg == -1:
        return "판정불가"
    return "긍정" if pos > neg else "부정"


def evaluate(name, predictions):
    """예측 결과를 정답과 비교해 정확도를 출력한다."""
    correct = sum(p == r["label"] for p, r in zip(predictions, reviews))
    acc = correct / len(reviews) * 100
    print(f"\n[{name}] 정확도 : {correct}/{len(reviews)} = {acc:.0f}%")
    return acc


# %% [markdown]
# ## 기법 1 — Zero-shot
#
# 예시 없이 작업 지시만 줍니다. 이후 모든 기법의 비교 기준선(baseline)입니다.  
#
# **관찰 포인트**: 아무 도움 없이 모델이 어디까지 맞히는가.

# %%
zero_shot_system = (
    "당신은 영화 리뷰 감성 분석기입니다. "
    "리뷰를 읽고 '긍정' 또는 '부정' 둘 중 하나의 단어로만 답하세요."
)

zero_shot_preds = []
for r in reviews:
    out = generate_response(zero_shot_system, r["text"], max_new_tokens=10)
    pred = parse_label(out)
    zero_shot_preds.append(pred)
    mark = "O" if pred == r["label"] else "X"
    print(f"[{mark}] 예측={pred} (정답={r['label']}) | 출력='{out}'")

zero_shot_acc = evaluate("Zero-shot", zero_shot_preds)

# %% [markdown]
# ## 기법 2 — Few-shot
#
# 정답 예시 3개를 먼저 보여준 뒤, 같은 형식으로 답하게 합니다.
# 세 번째 예시는 일부러 반어·역접 패턴을 넣어 함정 리뷰 대비를 돕습니다.  
#
# **관찰 포인트**: 예시가 정확도와 출력 형식 일관성을 얼마나 올리는가.

# %%
few_shot_system = (
    "당신은 영화 리뷰 감성 분석기입니다. "
    "아래 예시와 동일한 형식으로 '긍정' 또는 '부정' 한 단어만 답하세요."
)

# 모델에게 보여줄 정답 예시 3개
few_shot_examples = """리뷰: 연출과 연기 모두 완벽했다. 강력 추천한다.
감성: 긍정

리뷰: 돈과 시간이 아까운 영화. 두 번은 못 본다.
감성: 부정

리뷰: 겉보기엔 화려하다고 칭찬하고 싶지만, 사실 보는 내내 지루했다.
감성: 부정
"""

few_shot_preds = []
for r in reviews:
    # 예시 뒤에 분류할 리뷰를 같은 형식으로 이어 붙인다
    user_msg = few_shot_examples + f"\n리뷰: {r['text']}\n감성:"
    out = generate_response(few_shot_system, user_msg, max_new_tokens=10)
    pred = parse_label(out)
    few_shot_preds.append(pred)
    mark = "O" if pred == r["label"] else "X"
    print(f"[{mark}] 예측={pred} (정답={r['label']}) | 출력='{out}'")

few_shot_acc = evaluate("Few-shot", few_shot_preds)

# %% [markdown]
# ## 기법 3 — CoT (Chain-of-Thought)
#
# 결론을 바로 내리지 말고, 감정 표현을 **단계적으로 분석한 뒤** 마지막에 결론을 적게 합니다. 반어법 같은 함정에서 특히 효과가 큽니다.  
#
# **관찰 포인트**: 추론 과정을 노출시키면 오류가 줄어드는가.

# %%
cot_system = (
    "당신은 영화 리뷰 감성 분석기입니다. 다음 순서로 답하세요.\n"
    "1) 리뷰에서 감정을 드러내는 표현을 찾아 단계적으로 분석한다.\n"
    "2) 반어법·역접 표현이 있으면 글쓴이의 실제 의도를 짚는다.\n"
    "3) 마지막 줄에 반드시 '결론: 긍정' 또는 '결론: 부정' 형식으로 적는다."
)

cot_preds = []
for r in reviews:
    out = generate_response(cot_system, r["text"], max_new_tokens=300)
    pred = parse_label(out)
    cot_preds.append(pred)
    mark = "O" if pred == r["label"] else "X"
    print(f"\n[{mark}] 정답={r['label']} 예측={pred}")
    print(f"리뷰 : {r['text']}")
    print(f"추론 :\n{out}")

cot_acc = evaluate("CoT", cot_preds)

# %% [markdown]
# ## 기법 4 — Self-Consistency
#
# CoT prompt 를 **그대로 5회 호출**하되, 매번 다른 추론 경로가 나오도록 샘플링을
# 켭니다. 5개 결론을 모아 **다수결**로 최종 답을 정합니다.  
#
# **관찰 포인트**: 한 번 호출에서 생기는 단발성 오류를 걸러낼 수 있는가.

# %%
from collections import Counter

def self_consistency(review_text, n_samples=5):
    """동일한 CoT prompt 를 n_samples 회 샘플링 호출한 뒤 다수결로 결정한다."""
    votes = []
    for _ in range(n_samples):
        out = generate_response(cot_system, review_text, max_new_tokens=300,
                                do_sample=True, temperature=0.9)
        votes.append(parse_label(out))
    # 가장 많이 나온 라벨을 최종 답으로 선택
    final = Counter(votes).most_common(1)[0][0]
    return final, votes


sc_preds = []
for r in reviews:
    final, votes = self_consistency(r["text"], n_samples=5)
    sc_preds.append(final)
    mark = "O" if final == r["label"] else "X"
    print(f"[{mark}] 정답={r['label']} | 5회 투표={votes} -> 다수결={final}")

sc_acc = evaluate("Self-Consistency", sc_preds)

# %% [markdown]
# ## 기법 5 — Structured Output
#
# 출력을 자유 텍스트가 아니라 **고정된 JSON 스키마**로 받습니다.
# `Pydantic` 모델이 타입·값 범위·필수 필드를 자동 검증하므로, 후처리 코드 없이
# 바로 프로그램에서 사용할 수 있습니다.  
#
# **관찰 포인트**: 감성·확신도·근거 키워드를 한 번에, 안정적으로 받을 수 있는가.
#
# > HyperCLOVAX 같은 소형 로컬 모델은 JSON 형식을 가끔 어기므로, 파싱 실패에
# > 대비한 `try/except` 가 필요합니다. 이것이 OpenAI 의 native structured output
# > 기능과의 차이입니다.

# %%
import json
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class ReviewAnalysis(BaseModel):
    """리뷰 분석 결과 스키마 — sentiment / confidence / keywords"""
    sentiment: Literal["긍정", "부정"]
    confidence: float = Field(ge=0.0, le=1.0, description="0~1 사이 확신도")
    keywords: list[str] = Field(description="그렇게 판단한 근거 단어들")

structured_system = (
    "당신은 영화 리뷰 감성 분석기입니다. 리뷰를 분석해 아래 JSON 형식으로만 답하세요. "
    "JSON 외의 설명·인사말은 절대 쓰지 마세요.\n"
    '{"sentiment": "긍정 또는 부정", '
    '"confidence": 0과 1 사이 숫자, '
    '"keywords": ["판단 근거가 된 단어", "..."]}'
)

def extract_json(text):
    """모델 출력에서 첫 번째 중괄호 JSON 블록을 추출한다."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else None

structured_preds = []
for r in reviews:
    raw = generate_response(structured_system, r["text"], max_new_tokens=200)
    json_str = extract_json(raw)
    try:
        # Pydantic 이 타입·값 범위·필수 필드를 자동 검증한다
        analysis = ReviewAnalysis.model_validate_json(json_str)
        structured_preds.append(analysis.sentiment)
        mark = "O" if analysis.sentiment == r["label"] else "X"
        print(f"[{mark}] 정답={r['label']} | {analysis.model_dump()}")
    except (ValidationError, TypeError, json.JSONDecodeError):
        # JSON 형식을 어겼거나 비어 있는 경우
        structured_preds.append("판정불가")
        print(f"[X] 정답={r['label']} | 파싱 실패 -> 원본='{raw[:80]}'")

structured_acc = evaluate("Structured Output", structured_preds)

# %% [markdown]
# ## 정리 — 5가지 기법 정확도 비교

# %%
print("=" * 38)
print(f"  {'기법':<20}{'정확도':>8}")
print("-" * 38)
for name, acc in [
    ("1. Zero-shot", zero_shot_acc),
    ("2. Few-shot", few_shot_acc),
    ("3. CoT", cot_acc),
    ("4. Self-Consistency", sc_acc),
    ("5. Structured Output", structured_acc),
]:
    print(f"  {name:<22}{acc:>5.0f}%")
print("=" * 38)

# %% [markdown]
# ## 실습 과제
#
# 1. `reviews` 리스트에 직접 고른 영화 리뷰 5개를 추가하고 정답 라벨을 매겨,
#    어떤 기법이 가장 견고한지 다시 비교해 보세요.
# 2. Few-shot 의 예시 개수를 1개 / 3개 / 5개로 바꿔 가며 정확도 변화를 관찰하세요.
# 3. CoT 의 `cot_system` 지시문을 바꿔(예: 분석 단계 추가) 추론 품질이 달라지는지 보세요.
# 4. Self-Consistency 의 `n_samples` 와 `temperature` 를 조정하며 안정성 변화를 확인하세요.
# 5. `ReviewAnalysis` 스키마에 `summary`(한 줄 요약) 필드를 추가하고 prompt 도 함께 수정해 보세요.

# %%
