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
# .env 파일의 HF_TOKEN 을 환경변수로 불러온다 (토큰을 코드에 직접 노출하지 않기 위함)
#load_dotenv()
#login(token=os.environ["HF_TOKEN"])

# %%

# %% [markdown]
# ## 1. 공통 데이터 — 영화 리뷰 + 정답 라벨
#
# 5개 리뷰 중 마지막 문장은 **반어법**(칭찬하는 듯하지만 실제로는 혹평)이라,
# 기법별 성능 차이를 드러내는 함정 역할을 합니다.

# %%

# %% [markdown]
# ## 2. 공통 헬퍼 함수
#
# 모든 기법이 같은 모델 호출 함수를 사용합니다. Self-Consistency 에서 매 호출마다
# 다른 결과가 나오도록 `do_sample=True`(확률적 샘플링) 옵션도 지원합니다.

# %%
def generate_response(system_content, user_content, max_new_tokens=512,
        # 샘플링 활성화 — 같은 prompt 라도 호출마다 다른 추론 경로가 나온다
    # 입력 프롬프트를 제외하고 새로 생성된 토큰만 디코딩
    # HyperCLOVAX 종료 토큰이 남아 있으면 잘라낸다


# %%
def parse_label(text):
def evaluate(name, predictions):

# %% [markdown]
# ## 기법 1 — Zero-shot
#
# 예시 없이 작업 지시만 줍니다. 이후 모든 기법의 비교 기준선(baseline)입니다.  
#
# **관찰 포인트**: 아무 도움 없이 모델이 어디까지 맞히는가.

# %%

# %% [markdown]
# ## 기법 2 — Few-shot
#
# 정답 예시 3개를 먼저 보여준 뒤, 같은 형식으로 답하게 합니다.
# 세 번째 예시는 일부러 반어·역접 패턴을 넣어 함정 리뷰 대비를 돕습니다.  
#
# **관찰 포인트**: 예시가 정확도와 출력 형식 일관성을 얼마나 올리는가.

# %%
# 모델에게 보여줄 정답 예시 3개
    # 예시 뒤에 분류할 리뷰를 같은 형식으로 이어 붙인다


# %% [markdown]
# ## 기법 3 — CoT (Chain-of-Thought)
#
# 결론을 바로 내리지 말고, 감정 표현을 **단계적으로 분석한 뒤** 마지막에 결론을 적게 합니다. 반어법 같은 함정에서 특히 효과가 큽니다.  
#
# **관찰 포인트**: 추론 과정을 노출시키면 오류가 줄어드는가.

# %%

# %% [markdown]
# ## 기법 4 — Self-Consistency
#
# CoT prompt 를 **그대로 5회 호출**하되, 매번 다른 추론 경로가 나오도록 샘플링을
# 켭니다. 5개 결론을 모아 **다수결**로 최종 답을 정합니다.  
#
# **관찰 포인트**: 한 번 호출에서 생기는 단발성 오류를 걸러낼 수 있는가.

# %%
def self_consistency(review_text, n_samples=5):
    # 가장 많이 나온 라벨을 최종 답으로 선택


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
class ReviewAnalysis(BaseModel):
def extract_json(text):
        # Pydantic 이 타입·값 범위·필수 필드를 자동 검증한다
        # JSON 형식을 어겼거나 비어 있는 경우

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
# 5. `ReviewAnalysis` 스키마에 `summary`(한 줄 요약) 필드를 추가하고 prompt 도 함께 수정해 보세요.

# %%
