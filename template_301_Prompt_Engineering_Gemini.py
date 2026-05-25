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
# # 실습 — 5가지 prompt 기법 비교 (Gemini 버전)
#
# **공통 태스크: 영화 리뷰의 감성 분류 + 그렇게 판단한 근거 키워드 추출**
#
# 동일한 영화 리뷰 데이터에 5가지 prompt 기법을 차례로 적용하고, 기법이 바뀔 때
# 정확도와 출력 형태가 어떻게 달라지는지 비교합니다. 모델은 Google 의
# `gemini-2.5-flash-lite` API 를 사용합니다.
# (300 노트북은 동일한 실습을 로컬 HyperCLOVAX 모델로 수행합니다.)
#
# | 기법 | Prompt 핵심 | 관찰 포인트 |
# |------|-------------|-------------|
# | Zero-shot | 이 리뷰의 감성을 분류해줘 | 기본 정확도 측정 (baseline) |
# | Few-shot | 예시 3개를 보여준 뒤 동일 형식으로 답하게 함 | 예시가 정확도·일관성을 얼마나 올리는가 |
# | CoT | 단계적으로 판단 근거를 설명한 후 결론 | 추론 노출이 오류를 줄이는가 |
# | Self-Consistency | 동일 CoT prompt 를 여러 번(여기선 3회) 호출 후 다수결 | 단발성 오류 제거 효과 |
# | Structured Output | Pydantic 스키마로 sentiment / confidence / keywords 반환 | 후처리 코드 없이 바로 사용 가능한가 |
#
# > Gemini API 는 Colab GPU 가 필요 없습니다. API 키만 있으면 어디서든 실행됩니다.

# %% [markdown]
# ## 0. 환경 설정
#
# - Google AI Studio 에서 API 키 발급: https://aistudio.google.com/app/apikey
# - 발급받은 키를 코드에 직접 쓰지 말고 프로젝트 루트의 `.env` 파일에 저장합니다.
#   `.env` 는 `.gitignore` 에 포함되어 git 에 커밋되지 않습니다.
#
# ```
# GOOGLE_API_KEY=AIza...
# ```

# %%
# .env 파일의 GOOGLE_API_KEY 를 불러와 Gemini API 클라이언트를 생성한다

# %% [markdown]
# ## 1. 공통 데이터 — 영화 리뷰 + 정답 라벨
#
# 3개 리뷰 중 마지막 문장은 **반어법**(칭찬하는 듯하지만 실제로는 혹평)이라,
# 기법별 성능 차이를 드러내는 함정 역할을 합니다.
# (Gemini 무료 API 호출 수를 줄이려고 리뷰를 3개로 구성했습니다 — 필요하면 더 추가하세요.)

# %%

# %% [markdown]
# ## 2. 공통 헬퍼 함수
#
# 모든 기법이 같은 Gemini 호출 함수를 사용합니다. Self-Consistency 에서 매 호출마다
# 다른 결과가 나오도록 `do_sample=True`(temperature 적용) 옵션도 지원합니다.
#
# > **무료 등급 주의**: Gemini 무료 API 는 분당·일일 요청 수 제한이 낮습니다. 그래서
# > (1) `_throttle` 로 호출 사이에 최소 간격을 두어 분당 제한(RPM)을 피하고,
# > (2) 그래도 429(쿼터 초과)가 나면 `_generate_with_retry` 가 안내된 시간만큼
# > 대기 후 재시도합니다. 이 노트북은 리뷰 3개·Self-Consistency 3회로 호출 수를
# > 약 21회로 줄여 무료 등급 한도 안에서 한 번에 완주하도록 구성했습니다.
# > 재시도를 모두 소진하며 계속 429 가 난다면 **일일 할당량 소진**입니다 —
# > 태평양 시간 자정에 리셋됩니다. 참고: https://ai.google.dev/gemini-api/docs/rate-limits

# %%
# 호출 사이 최소 간격(초). 무료 등급 분당 제한(RPM)에 걸리지 않도록 띄운다.
# gemini-2.5-flash-lite 는 무료 RPM 이 비교적 높아 5초로 설정. 계속 429 가 나면 키우세요.
def _throttle():
def _generate_with_retry(config, content, max_retries=6):
            # 429(RESOURCE_EXHAUSTED) 외의 오류는 재시도해도 소용없으므로 그대로 올린다
                # 재시도를 모두 소진 — 대개 무료 등급 일일 할당량 소진이 원인이다
            # 429 응답의 RetryInfo 에 권장 대기 시간이 'retryDelay': '13s' 형태로 들어 있다
def generate_response(system_content, user_content, do_sample=False, temperature=0.8):
    # system_instruction 으로 역할(시스템 메시지)을 지정한다


# %%
def parse_label(text):
def evaluate(name, predictions):

# %% [markdown]
# ## 기법 1 — Zero-shot
#
# 예시 없이 작업 지시만 줍니다. 이후 모든 기법의 비교 기준선(baseline)입니다.
# **관찰 포인트**: 아무 도움 없이 모델이 어디까지 맞히는가.

# %%

# %% [markdown]
# ## 기법 2 — Few-shot
#
# 정답 예시 3개를 먼저 보여준 뒤, 같은 형식으로 답하게 합니다.
# 세 번째 예시는 일부러 반어·역접 패턴을 넣어 함정 리뷰 대비를 돕습니다.
# **관찰 포인트**: 예시가 정확도와 출력 형식 일관성을 얼마나 올리는가.

# %%
# 모델에게 보여줄 정답 예시 3개
    # 예시 뒤에 분류할 리뷰를 같은 형식으로 이어 붙인다


# %% [markdown]
# ## 기법 3 — CoT (Chain-of-Thought)
#
# 결론을 바로 내리지 말고, 감정 표현을 **단계적으로 분석한 뒤** 마지막에 결론을
# 적게 합니다. 반어법 같은 함정에서 특히 효과가 큽니다.
# **관찰 포인트**: 추론 과정을 노출시키면 오류가 줄어드는가.

# %%

# %% [markdown]
# ## 기법 4 — Self-Consistency
#
# CoT prompt 를 **그대로 여러 번 호출**(여기서는 3회)하되, 매번 다른 추론 경로가
# 나오도록 temperature 를 높입니다. 결론들을 모아 **다수결**로 최종 답을 정합니다.
# (이론적으로는 5회 이상이 좋지만, 무료 API 호출 수를 줄이려고 3회로 설정했습니다.)
# **관찰 포인트**: 한 번 호출에서 생기는 단발성 오류를 걸러낼 수 있는가.

# %%
def self_consistency(review_text, n_samples=3):
    # 가장 많이 나온 라벨을 최종 답으로 선택


# %% [markdown]
# ## 기법 5 — Structured Output
#
# 출력을 자유 텍스트가 아니라 **고정된 JSON 스키마**로 받습니다.
# Gemini 는 **native structured output** 을 지원해, `response_schema` 에 Pydantic
# 모델을 직접 넘기면 모델이 그 스키마를 강제로 따르는 JSON 을 생성합니다.
# **관찰 포인트**: 감성·확신도·근거 키워드를 한 번에, 안정적으로 받을 수 있는가.
#
# > 300(CLOVAX) 노트북은 로컬 소형 모델이라 JSON 형식을 가끔 어겨 `try/except` 가
# > 필요했지만, Gemini 는 `response_schema` 로 형식이 보장되므로 후처리 코드 없이
# > 바로 사용할 수 있습니다. — 이것이 native structured output 의 장점입니다.
# >
# > 단, `confidence` 의 0~1 범위는 Pydantic 의 `ge`/`le` 제약 대신 필드
# > `description` 으로만 안내합니다 — 숫자 범위(min/max) 제약은 API 스키마 변환
# > 과정에서 누락될 수 있어, 범위는 description 으로 알리고 검증은 Pydantic 이 맡습니다.

# %%
class ReviewAnalysis(BaseModel):
    # 참고: 0~1 범위는 ge·le 제약 대신 description 으로 안내한다.
    #       범위 제약(min/max)은 API 스키마 변환에서 누락될 수 있어 검증은 Pydantic 이 맡는다.
# response_schema 에 Pydantic 모델을 넘기면 Gemini 가 해당 스키마를 강제로 따른다
    # 구조화 출력 호출도 동일하게 429 자동 재시도 로직을 거친다
    # json_text 는 스키마를 따르는 JSON 문자열 — Pydantic 으로 바로 검증

# %% [markdown]
# ## 정리 — 5가지 기법 정확도 비교

# %%

# %% [markdown]
# ## 실습 과제
#
# 1. `reviews` 리스트에 직접 고른 영화 리뷰 5개를 추가하고 정답 라벨을 매겨,
#    어떤 기법이 가장 견고한지 다시 비교해 보세요.
# 2. Few-shot 의 예시 개수를 1개 / 3개 / 5개로 바꿔 가며 정확도 변화를 관찰하세요.
# 3. CoT 의 `cot_system` 지시문을 바꿔(예: 분석 단계 추가) 추론 품질이 달라지는지 보세요.
# 4. Self-Consistency 의 `n_samples` 와 `temperature` 를 조정하며 안정성 변화를 확인하세요.
# 5. `ReviewAnalysis` 스키마에 `summary`(한 줄 요약) 필드를 추가해 보세요.
#    Gemini 는 `response_schema` 만 바꾸면 prompt 수정 없이도 새 필드를 채워 줍니다.
# 6. 같은 실습을 로컬 모델로 수행하는 300(CLOVAX) 노트북과 결과를 비교해 보세요.

# %%
