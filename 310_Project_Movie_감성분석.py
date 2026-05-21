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
# # 딥러닝 기반 영화 리뷰 감성 분석 시스템
#
# ## 프로젝트 목표
# - 딥러닝 기법을 활용한 감성 분석 시스템 구축
# - 두 가지 방법 비교: Hugging Face Pipeline vs ClovaX
# - 네이버 영화평 데이터를 이용한 실전 감성 분석
#
# ## 학습 내용
# 1. 최소한의 데이터 전처리
# 2. 감성 분석 방법 1: Hugging Face Pipeline
# 3. 감성 분석 방법 2: ClovaX

# %% [markdown]
# ---
# ## 1. 데이터 준비 및 최소 전처리

# %%
import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
import os
from huggingface_hub import login
warnings.filterwarnings('ignore')

# 환경 변수 로드
load_dotenv()

token = os.getenv("HF_TOKEN")
if token:
    login(token=token)

# %%
# 네이버 영화평 데이터 다운로드
DATA_TRAIN_PATH = tf.keras.utils.get_file(
    "ratings_train.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_train.txt"
)
DATA_TEST_PATH = tf.keras.utils.get_file(
    "ratings_test.txt",
    "https://raw.github.com/ironmanciti/Infran_NLP/master/data/naver_movie/ratings_test.txt"
)

# 데이터 로드
train_data = pd.read_csv(DATA_TRAIN_PATH, delimiter='\t')
test_data = pd.read_csv(DATA_TEST_PATH, delimiter='\t')

print("=" * 80)
print("[데이터 로드 완료]")
print("=" * 80)
print(f"훈련 데이터: {train_data.shape}")
print(f"테스트 데이터: {test_data.shape}")

# 결측값 제거
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)

# 데이터 샘플링
df_train = train_data.sample(n=5_000, random_state=1)
df_test = test_data.sample(n=1_000, random_state=1)

print(f"\n샘플링 후:")
print(f"훈련 데이터: {df_train.shape}")
print(f"테스트 데이터: {df_test.shape}")

# 딥러닝 모델용 최소 전처리 (결측값 제거만)
df_train['cleaned_document'] = df_train['document'].astype(str)
df_test['cleaned_document'] = df_test['document'].astype(str)

# 빈 문자열 제거
df_train = df_train[df_train['cleaned_document'].str.len() > 0]
df_test = df_test[df_test['cleaned_document'].str.len() > 0]

print(f"\n전처리 후:")
print(f"훈련 데이터: {df_train.shape}")
print(f"테스트 데이터: {df_test.shape}")

# %% [markdown]
# ---
# ## 2. 감성 분석 방법 1: Hugging Face Pipeline

# %% [markdown]
# ### 2.1 파이프라인을 이용한 감성 분석

# %%
# 다국어 감성 분석 모델
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_classifier = pipeline('sentiment-analysis', model=model_name)

# 샘플 리뷰 분석
sample_texts = [
    "다시는 보고 싶지 않은 짜증나는 영화",
    "아주 재미있는 영화",
    "정말 재미없는 영화였다",
    "이 영화 최고",
    "보통 영화"
]

print("\n[감성 분석 결과]")
results = sentiment_classifier(sample_texts)

for i, result in enumerate(results):
    print(f"{sample_texts[i]}")
    print(f"  → {result['label']}, 신뢰도: {result['score']:.4f}\n")

# %% [markdown]
# ### 2.2 테스트 데이터 감성 분석

# %%
# 테스트 데이터 샘플 분석
test_samples = df_test['cleaned_document'].head(10).tolist()
test_labels = df_test['label'].head(10).tolist()

print("\n[테스트 데이터 샘플 분석]")
sentiment_results = sentiment_classifier(test_samples)

correct = 0
for i, (text, true_label, result) in enumerate(zip(test_samples, test_labels, sentiment_results)):
    # 별점을 긍정/부정으로 변환 (4-5점: 긍정, 1-2점: 부정)
    predicted_label = 1 if int(result['label'].split()[0]) >= 4 else 0
    true_label_str = "긍정" if true_label == 1 else "부정"
    predicted_label_str = "긍정" if predicted_label == 1 else "부정"

    is_correct = "✓" if true_label == predicted_label else "✗"
    if true_label == predicted_label:
        correct += 1

    print(f"\n리뷰 {i+1}: {text[:50]}...")
    print(f"  실제: {true_label_str}, 예측: {predicted_label_str} ({result['label']}) {is_correct}")

accuracy_hf = correct / len(test_samples)
print(f"\n정확도: {accuracy_hf:.4f} ({correct}/{len(test_samples)})")

# %% [markdown]
# ---
# ## 3. 감성 분석 방법 2: ClovaX

# %% [markdown]
# ### 3.1 ClovaX를 사용한 감성 분석
#
# **지시사항**:
# - ClovaX 모델을 사용하여 감성 분석 수행
# - 프롬프트를 통해 긍정/부정 분류 요청
# - Hugging Face와 동일한 테스트 데이터로 평가
#
# %%
print("=" * 80)
print("[ClovaX 모델 로드 중...]")
print("=" * 80)

model_name = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
clovax_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
clovax_tokenizer = AutoTokenizer.from_pretrained(model_name)

print("✓ ClovaX 모델 로드 완료")


# %%
def analyze_sentiment_clovax(text):
    """
    ClovaX 모델을 사용한 감성 분석

    Args:
        text: 분석할 텍스트

    Returns:
        감성 분석 결과 (긍정/부정)
    """
    system_content = "당신은 영화 리뷰의 감성을 분석하는 전문가입니다. 주어진 리뷰를 긍정 또는 부정으로 분류해주세요."
    user_content = f"""다음 영화 리뷰의 감성을 분석하여 긍정 또는 부정으로 분류해주세요.

리뷰: {text}

답변 형식: 긍정 또는 부정만 출력하세요."""

    chat = [
        {"role": "tool_list", "content": ""},
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

    inputs = clovax_tokenizer.apply_chat_template(
        chat,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = inputs.to(clovax_model.device)

    output_ids = clovax_model.generate(
        **inputs,
        max_length=512,
        repetition_penalty=1.2,
        eos_token_id=clovax_tokenizer.eos_token_id,
    )

    output_text = clovax_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

    # 필요시 <|endofturn|>, <|stop|> 등에서 자르기
    for stop_str in ["<|endofturn|>", "<|stop|>"]:
        if stop_str in output_text:
            output_text = output_text.split(stop_str)[0]

    # 생성된 텍스트에서 사용자 입력 부분 제거
    if user_content in output_text:
        result_text = output_text.split(user_content)[-1].strip()
    else:
        result_text = output_text.strip()

    # "긍정" 또는 "부정" 키워드 확인
    if "긍정" in result_text:
        return 1
    elif "부정" in result_text:
        return 0
    else:
        # 기본값 (긍정으로 가정)
        return 1

# 샘플 리뷰 분석
sample_texts = [
    "다시는 보고 싶지 않은 짜증나는 영화",
    "아주 재미있는 영화",
    "정말 재미없는 영화였다",
    "이 영화 최고",
    "보통 영화"
]

print("\n[감성 분석 결과]")
for text in sample_texts:
    result = analyze_sentiment_clovax(text)
    sentiment = "긍정" if result == 1 else "부정"
    print(f"{text} → {sentiment}")

# %% [markdown]
#    ### 3.2 테스트 데이터 감성 분석
# %%
print("=" * 80)
print("[테스트 데이터 감성 분석 (ClovaX)]")
print("=" * 80)

# 테스트 데이터 샘플 분석
test_samples = df_test['cleaned_document'].head(10).tolist()
test_labels = df_test['label'].head(10).tolist()

print("\n[테스트 데이터 샘플 분석]")
clovax_results = []

correct_clovax = 0
for i, (text, true_label) in enumerate(zip(test_samples, test_labels)):
    predicted_label = analyze_sentiment_clovax(text)

    if predicted_label is not None:
        true_label_str = "긍정" if true_label == 1 else "부정"
        predicted_label_str = "긍정" if predicted_label == 1 else "부정"

        is_correct = "✓" if true_label == predicted_label else "✗"
        if true_label == predicted_label:
            correct_clovax += 1

        print(f"\n리뷰 {i+1}: {text[:50]}...")
        print(f"  실제: {true_label_str}, 예측: {predicted_label_str} {is_correct}")

        clovax_results.append(predicted_label)
    else:
        clovax_results.append(None)

accuracy_clovax = correct_clovax / len(test_samples)
print(f"\n정확도: {accuracy_clovax:.4f} ({correct_clovax}/{len(test_samples)})")

# %%

