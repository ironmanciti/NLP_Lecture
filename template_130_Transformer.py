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
# # 130. Neural machine translation with a Transformer and Keras

# %% [markdown]
# 이 튜토리얼은 Transformer 모델 을 학습시켜 포르투갈어를 영어 데이터 세트 로 번역합니다. 이것은 텍스트 생성 및 Attention 에 대한 지식을 가정한 고급 예제입니다.
#
# Transformer 모델의 핵심 아이디어는 self-attention (입력 시퀀스의 표현을 계산하기 위해 입력 시퀀스의 다른 위치에 주의를 기울이는 기능)입니다. Transformer는 self-attention 레이어의 스택을 생성하고 아래의 Scaled dot product Attention 및 Multi-head Attention 섹션에서 설명됩니다.

# %% [markdown]
# <img src="https://www.tensorflow.org/images/tutorials/transformer/apply_the_transformer_to_machine_translation.gif" alt="Applying the Transformer to machine translation">
#
# Figure 1:Transformer를 기계 번역에 적용합니다.
#

# %% [markdown]
# 이 튜토리얼에서는 다음을 수행합니다.
#
# - 데이터 준비  
# - 필요한 구성요소 구현:  
#    - 위치 임베딩.  
#    - Attention layer.  
#    - 인코더와 디코더.  
# - Transformer를 제작 및 훈련  .
# - 번역 생성  

# %% [markdown]
# 이 튜토리얼에서는  4층 Transformer를 구축합니다.

# %% [markdown]
# 이 노트북에서 모델을 훈련한 후에는 포르투갈어 문장을 입력하고 영어 번역을 반환할 수 있습니다.
#
# <img src="https://www.tensorflow.org/images/tutorials/transformer/attention_map_portuguese.png" alt="Attention heatmap">
#
# Figure 2: 이 튜토리얼이 끝나면 생성할 수 있는 시각화된 Attention 가중치.

# %% [markdown]
# ## Setup

# %%
# 최신 버전의 TensorFlow를 설치하여
# `tf.keras.layers.MultiHeadAttention`의 개선된 마스킹 지원을 사용하세요.
# Hugging Face datasets 추가 설치 (OPUS-100 코퍼스 로드용)

# %%
# GPU 사용 가능 여부 확인
    # GPU 메모리 증가 설정 (필요한 만큼만 사용)


# %% [markdown]
# ## 데이터 처리
#
# 이 섹션에서는 데이터 세트와 하위 단어 토크나이저를 다운로드하고 훈련을 위해 `tf.data.Dataset`에 래핑합니다.

# %% [markdown]
# ### Download the dataset

# %% [markdown]
# ### 데이터셋 로드
#
# > **⚠️ 원 튜토리얼 데이터셋 대체 안내**
# >
# > 원 튜토리얼에서 사용하는 `ted_hrlr_translate/pt_to_en` 데이터셋은 **원본 호스팅 서버(phontron.com)의 다운로드 중단**으로 더 이상 `tfds.load()`로 받을 수 없습니다 (`NonMatchingChecksumError`).
# >
# > 따라서 동일한 목적(포르투갈어 → 영어 번역)의 대체 코퍼스로 **Helsinki-NLP의 OPUS-100**을 사용합니다.
# >
# > - ✅ Hugging Face가 **Parquet 네이티브**로 직접 호스팅 → 외부 URL 문제 없음
# > - ✅ 약 **100만 개** 훈련 쌍, 2,000개 검증/테스트 쌍 (원본보다 훨씬 큰 규모)
# > - ✅ 수업 진행 속도를 위해 원 튜토리얼 규모(~52,000 훈련 쌍)로 서브샘플링
# >
# > **중요**: 출력 인터페이스를 TFDS와 동일한 `(pt_string, en_string)` 튜플 `tf.data.Dataset`으로 맞췄기 때문에, **이후 모든 셀(토크나이저, 배치, 트랜스포머 학습)은 수정 없이 그대로 작동**합니다.
#

# %%
# ─────────────────────────────────────────────────────────────
# OPUS-100 en-pt 병렬 코퍼스 로드 (TFDS 대체)
# ─────────────────────────────────────────────────────────────
# 1) Hugging Face에서 parquet 파일 직접 로드 (외부 URL 의존 없음)
# DatasetDict({
#     train:      Dataset({ features: ['translation'], num_rows: 1000000 })
#     validation: Dataset({ features: ['translation'], num_rows: 2000 })
#     test:       Dataset({ features: ['translation'], num_rows: 2000 })
# })
# 2) 데이터 구조 확인
# {'translation': {'en': '...', 'pt': '...'}}
# 3) 수업용 규모로 서브샘플링
#    (원 ted_hrlr pt_to_en 튜토리얼의 train ~52k, val ~1.2k 규모에 맞춤)
# 4) HuggingFace Dataset → tf.data.Dataset 변환
#    핵심: TFDS의 as_supervised=True 출력과 동일하게
#    (pt_string_tensor, en_string_tensor) 튜플을 yield 하도록 구성
#    → 이후 모든 셀에서 train_examples.batch(...), tokenizers.pt.tokenize(...) 등이
#      기존 코드 수정 없이 그대로 동작
def hf_to_tf_dataset(hf_split):

# %% [markdown]
# TensorFlow Datasets에서 반환된 `tf.data.Dataset` 객체는 텍스트 예제 쌍을 생성합니다.

# %%
# 훈련 데이터셋에서 포르투갈어 및 영어 예제 출력

# %% [markdown]
# ### Tokenizer Set up

# %% [markdown]
# 이제 데이터세트를 로드했으므로 각 요소가 토큰 ID(a)로 표시되도록 텍스트를 토큰화해야 합니다.

# %% [markdown]
# 이 튜토리얼에서는 [서브워드 토크나이저](https://www.tensorflow.org/text/guide/subwords_tokenizer) 튜토리얼에 내장된 토크나이저를 사용합니다.

# %%
# 'ted_hrlr_translate_pt_en_converter' 모델 다운로드 및 압축 해제

# %%

# %%
# 저장된 모델 로드

# %% [markdown]
# `tf.saved_model`에는 두 개의 텍스트 토크나이저가 포함되어 있습니다. 하나는 영어용이고 다른 하나는 포르투갈어용입니다. 둘 다 동일한 방법을 사용합니다.

# %%
# tokenizers.en 객체의 사용 가능한 메소드와 속성 리스트 출력

# %% [markdown]
# `tokenize` 메서드는 문자열 배치를 패딩된 토큰 ID 배치로 변환합니다. 이 방법은 토큰화하기 전에 입력을 구두점, 소문자로 나누고 유니코드 정규화합니다.

# %%
# TensorFlow 데이터셋에서 로드한 배치의 영어 문장 출력

# %%
# 로드된 토크나이저를 사용하여 영어 문장을 토큰 ID로 변환
# 패딩된 토큰 ID의 배치 출력

# %% [markdown]
# `detokenize` 메소드는 이러한 토큰 ID를 사람이 읽을 수 있는 텍스트로 다시 변환 합니다.

# %%
# 토큰 ID를 다시 사람이 읽을 수 있는 텍스트로 변환
# 변환된 텍스트 출력

# %% [markdown]
# 하위 수준 `lookup` 메서드는 토큰 ID를 토큰 텍스트로 변환합니다.

# %%
# 토큰 ID를 다시 개별 토큰으로 변환

# %% [markdown]
# 출력은 하위 단어 토큰화의 "subword" 측면을 보여줍니다.
#
# 예를 들어 'searchability''라는 단어는 'search''와 ''##ability''로 분해되고, 'serendipity''라는 단어는 ''s'', ''##ere'', ``##nd'`, ``##ip'` 및 ``##ity'`로 분해됩니다.
#
# 토큰화된 텍스트에는 `'[START]'` 및 `'[END]'` 토큰이 포함되어 있습니다.

# %% [markdown]
# "Transformer is awesome."을 tokenize 하면 다음과 같습니다.

# %%

# %% [markdown]
# 데이터 세트의 example당 토큰 분포는 다음과 같습니다.

# %%
# 포르투갈어와 영어 문장들의 토큰 개수를 저장할 리스트
# 훈련 데이터셋의 배치를 순회하며 토큰 개수 계산
  # 포르투갈어 문장 토큰화 및 토큰 개수 저장
  # 영어 문장 토큰화 및 토큰 개수 저장


# %%
# 모든 토큰 개수를 하나의 배열로 결합
# 히스토그램으로 토큰 개수 분포 시각화
# 가장 긴 문장의 토큰 개수 표시
# 그래프 제목 설정

# %% [markdown]
# ### `tf.data`를 사용하여 데이터 파이프라인 설정

# %% [markdown]
# 다음 함수는 텍스트 배치를 입력으로 사용하여 훈련에 적합한 형식으로 변환합니다.
#
# 1. 텍스트 배치를  ragged batch로 토큰화합니다.
# 2. `MAX_TOKENS`보다 길지 않도록 자릅니다.  
# 3. 대상(영어) 토큰을 입력과 레이블로 분할합니다. 이는 각 입력 위치에서 '레이블'이 다음 토큰의 ID가 되도록 한 단계씩 이동됩니다.  
# 4. `RaggedTensor`를 패딩된 dense `Tensor`로 변환합니다.  
# 5. `(inputs, labels)` 쌍을 반환합니다.  
#
# "Ragged batch"는 TensorFlow에서 사용되는 용어로, 각 요소의 길이가 서로 다른 배치를 의미합니다. 이는 특히 자연어 처리에서 문장 또는 문서의 길이가 서로 다를 때 이를 표현하기 위해 사용됩니다.
#  RaggedTensor는 이러한 가변 길이 시퀀스를 효율적으로 저장하고 처리할 수 있도록 설계되었습니다.

# %%

# %%
def prepare_batch(pt, en):
    # 포르투갈어 문장 토큰화 및 최대 토큰 수에 맞게 잘라내기
    # 영어 문장 토큰화 및 최대 토큰 수에 맞게 잘라내기


# %% [markdown]
# 아래 함수는 텍스트 예제 데이터 세트를 학습용 배치 데이터로 변환합니다.
#
# 1. 텍스트를 토큰화하고 너무 긴 시퀀스를 필터링합니다.  
# 2. `cache` 메소드는 해당 작업이 한 번만 실행되도록 보장합니다.  
# 3. 그런 다음 `shuffle` 및 `dense_to_ragged_batch`를 통해 순서를 무작위로 지정하고 예제 배치를 어셈블합니다.  
# 4. 마지막으로 'prefetch'는 모델과 병렬로 데이터세트를 실행하여 필요할 때 데이터를 사용할 수 있도록 합니다.

# %%

# %%
def make_batches(ds, num_samples=5000):


# %% [markdown]
#  </section>

# %% [markdown]
# ## Dataset 테스트

# %%

# %%
# 훈련 및 검증 데이터셋을 배치로 변환
# 학습 품질 향상을 위해 샘플 수 증가

# %% [markdown]
# Keras `Model.fit` train 에서는 `(입력, 레이블)` 쌍을 예상합니다.
# `입력`은 토큰화된 포르투갈어 및 영어 시퀀스 쌍 `(pt, en)`입니다.
# '레이블'은 1만큼 이동된 동일한 영어 시퀀스입니다.
# 이러한 변화는 각 위치에서 다음 토큰의 'label'인 'en' 시퀀스를 입력하도록 하기 위한 것입니다.

# %% [markdown]
# <table>
# <tr>
#   <th>Inputs at the bottom, labels at the top.</th>
# </tr>
# <tr>
#   <td>
#    <img width=400 src="https://www.tensorflow.org/images/tutorials/transformer/Transformer-1layer-words.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 이 설정은 각 단계의 모델 출력에 관계없이 다음 단계의 입력으로 실제 값을 얻기 때문에 "teacher forcing"이라고 합니다.
# 이는 텍스트 생성 모델을 교육하는 간단하고 효율적인 방법입니다.
# 모델을 순차적으로 실행할 필요가 없고, 서로 다른 시퀀스 위치의 출력을 병렬로 계산할 수 있으므로 효율적입니다.
#
# 포르투갈어 시퀀스가 주어지면 모델은 영어 시퀀스를 생성하려고 시도합니다.  
#
# 추론 루프를 작성하고 모델의 출력을 입력으로 다시 전달해야 합니다.  
# 모델은 훈련 중에 자체 오류를 수정하는 방법을 배워야 하기 때문에 보다 안정적인 모델을 제공할 수 있습니다.

# %%
# 훈련 배치의 첫 번째 요소 가져오기
# 포르투갈어 입력, 영어 입력, 영어 레이블의 shape 출력

# %% [markdown]
# `en`과 `en_labels`는 동일하며 한자리만 shift 했습니다.

# %%

# %% [markdown]
# ## 구성 요소 정의

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The original Transformer diagram</th>
#   <th colspan=1>A representation of a 4-layer Transformer</th>
# </tr>
# <tr>
#   <td>
#    <img width=400 src="https://www.tensorflow.org/images/tutorials/transformer/transformer.png"/>
#   </td>
#   <td>
#    <img width=307 src="https://www.tensorflow.org/images/tutorials/transformer/Transformer-4layer-compact.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# ### embedding 및 positional encoding layer

# %% [markdown]
# 인코더와 디코더에 대한 입력은 동일한 임베딩 및 위치 인코딩 logic을 사용합니다.
#
# <table>
# <tr>
#   <th colspan=1>The embedding and positional encoding layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/PositionalEmbedding.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 일련의 토큰이 주어지면 입력 토큰(포르투갈어)과 대상 토큰(영어) 모두 `tf.keras.layers.Embedding` 레이어를 사용하여 벡터로 변환되어야 합니다.
#
# 모델 전체에 사용되는 Attention 레이어는 입력을 순서가 없는 벡터 집합으로 간주합니다. 모델에는 순환 또는 컨벌루션 레이어가 포함되어 있지 않기 때문에 단어 순서를 식별할 수 있는 방법이 필요합니다.
#
# Transformer는 임베딩 벡터에 "위치 인코딩"을 추가합니다. 이는 (시퀀스 전반에 걸쳐) 서로 다른 주파수의 사인과 코사인 세트를 사용합니다.

# %% [markdown]
# 논문에서는 위치 인코딩을 계산하기 위해 다음 공식을 사용합니다.
#
# $$\Large{PE_{(pos, 2i)} = \sin(pos / 10000^{2i / d_{model}})} $$
# $$\Large{PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i / d_{model}})} $$

# %%
def positional_encoding(length, depth):
  # 각 위치에 대한 인덱스 생성
  # 각 깊이에 대한 각도 비율 계산
  # sin과 cos을 사용하여 위치 인코딩 생성
  # 결과를 TensorFlow 텐서로 변환


# %% [markdown]
# 위치 인코딩 기능은 임베딩 벡터의 깊이를 따라 위치에 대해 서로 다른 주파수에서 진동하는 사인과 코사인의 스택입니다. position 축을 따라 진동합니다.

# %%
# 위치 인코딩 생성
# 생성된 위치 인코딩의 형상 확인
# 위치 인코딩 시각화

# %% [markdown]
# 따라서 이를 사용하여 토큰의 임베딩 벡터를 조회하고 위치 벡터를 추가하는 `PositionEmbedding` 레이어를 만듭니다.

# %%
#positional_encoding 함수는 길이가 2048, 깊이가 512인 위치 인코딩 행렬을 생성합니다.
class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model):
  def compute_mask(self, *args, **kwargs):
    # 마스크 계산 (패딩된 부분을 모델이 무시하도록 함)
  def call(self, x):


# %%
# 포르투갈어와 영어를 위한 위치 인코딩 임베딩 레이어 생성
# 포르투갈어 입력에 대한 임베딩 적용
# 영어 입력에 대한 임베딩 적용

# %%
# 영어 임베딩 레이어의 마스크 속성 확인

# %% [markdown]
# ### Add and normalize
#
# <table>
# <tr>
#   <th colspan=2>Add and normalize</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/Add+Norm.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# "Add & Norm" 블록은 모델 전체에 분산되어 있습니다. 각각은 잔차 연결을 결합하고 `LayerNormalization` 레이어를 통해 결합한 결과를 실행합니다.
# 잔차 연결은 그래디언트에 대한 직접적인 경로를 제공하며 정규화는 출력에 대한 합리적인 scale을 유지합니다.

# %% [markdown]
# ### 기본 attention layer

# %% [markdown]
# Attention 레이어는 모델 전반에 걸쳐 사용됩니다. Attention이 구성되는 방식을 제외하고는 모두 동일합니다. 각각은 'layers.MultiHeadAttention', 'layers.LayerNormalization' 및 'layers.Add'를 포함합니다.
#
# <table>
# <tr>
#   <th colspan=2>The base attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/BaseAttention.png"/>
#   </td>
# </tr>
# </table>
#
# - MultiHeadAttention: 다중 헤드 어텐션 메커니즘을 구현하는 레이어입니다. 이 레이어는 시퀀스 내의 각 위치가 다른 위치의 정보를 어떻게 참조하는지를 학습합니다.  
# - LayerNormalization: 각 레이어의 출력을 정규화하는 레이어입니다. 이는 모델의 학습 안정성과 성능을 향상시킵니다.  
# - Add: 두 입력을 더하는 레이어로, 여기서는 잔차 연결(residual connection)을 구현하는 데 사용됩니다.

# %%
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    # 다중 헤드 어텐션(MultiHeadAttention) 레이어 초기화
    # 레이어 정규화(LayerNormalization) 레이어 초기화
    # 덧셈(Add) 레이어 초기화


# %% [markdown]
# #### Attention 작동 원리

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The base attention layer</th>
# </tr>
# <tr>
#   <td>
#    <img width=430 src="https://www.tensorflow.org/images/tutorials/transformer/BaseAttention-new.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 두 가지 입력이 있습니다.
#
# 1. query sequence; 처리 중인 시퀀스; attention을 하는 sequence(아래).
# 2. context sequence; attention 당하는 sequence(왼쪽).
#
# 출력은 쿼리 시퀀스와 동일한 모양을 갖습니다.
#
# 일반적인 비교는 이 작업이 dictionary 조회와 같다는 것입니다.
# 이 dictionary 조회는 '퍼지(fuzzy)'하고 '미분 가능(differentiable)'하며 '벡터화(vectorized)'된 형태의 사전 조회라고 할 수 있습니다.  
#  '퍼지'는 모호하거나 불확실한 정보를 처리할 수 있다는 의미이며, '미분 가능'은 연산이 최적화 과정에서 그래디언트를 통해 학습될 수 있음을 의미합니다. '벡터화'는 연산이 전체 시퀀스에 대해 동시에 수행되며, 각 요소가 벡터 형태의 데이터로 처리된다는 것을 나타냅니다.
#
# 다음은 단일 쿼리에 3개의 키와 3개의 값이 전달되는 일반 Python 사전입니다.
#
# ```
# d = {'color': 'blue', 'age': 22, 'type': 'pickup'}
# result = d['color']
# ```
#
# - 'query'는 당신이 찾으려는 것입니다.
# - 'key'는 사전이 어떤 정보를 가지고 있는지를 나타냅니다.
# - 'value'는 바로 그 정보입니다.
#
# 일반 dictionary에서 '쿼리'를 조회하면 dictionary는 일치하는 '키'를 찾아 관련 '값'을 반환합니다.  
# 키가 완벽하게 일치할 필요가 없는 **모호한(fuzzy)**한 dictionary를 상상할 수 있습니다.
# 위 사전에서 `d["species"]`를 검색했다면 검색어에 가장 잘 일치하는 `"pickup"`을 반환하기를 원할 수 있습니다.
#
# Attention 레이어는 이와 같은 퍼지 조회를 수행하지만 단지 최상의 키를 찾는 것이 아닙니다.
# '쿼리'가 각 '키'와 얼마나 잘 일치하는지에 따라 '값'을 결합합니다.
#
#  Attention 레이어에서 '쿼리', '키', '값'은 각각 벡터입니다.
# 해시 조회를 수행하는 대신 어텐션 레이어는 '쿼리'와 '키' 벡터를 결합하여 이들이 얼마나 잘 일치하는지, 즉 "어텐션 점수"를 결정합니다.
# 레이어는 'attention score'에 따라 가중치를 적용하여 모든 '값'에 대한 평균을 반환합니다.
#
# 쿼리 시퀀스의 각 위치는 '쿼리' 벡터를 제공합니다.
# 컨텍스트 시퀀스는 사전 역할을 합니다. 컨텍스트 시퀀스의 각 위치에는 '키' 및 '값' 벡터가 제공됩니다.
# 입력 벡터는 직접 사용되지 않습니다. 'layers.MultiHeadAttention' 레이어에는 입력 벡터를 사용하기 전에 투영(projection)하기 위한 'layers.Dense' 레이어가 포함되어 있습니다.
#

# %% [markdown]
# ### cross attention layer

# %% [markdown]
# Transformer의 문자 그대로 중심에는 Cross-Attention 레이어가 있습니다. 이 레이어는 인코더와 디코더를 연결합니다.
#
# <table>
# <tr>
#   <th colspan=1>The cross attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/CrossAttention.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 이를 구현하려면 `mha` 레이어를 호출할 때 대상 시퀀스 `x`를 `쿼리`로 전달하고 `컨텍스트` 시퀀스를 `key/value`로 전달합니다.

# %%
class CrossAttention(BaseAttention):
  def call(self, x, context):
    # 다중 헤드 어텐션을 쿼리, 키, 값에 적용
    # 어텐션 점수를 나중에 시각화하기 위해 저장
    # 어텐션 출력과 입력을 더하고, 레이어 정규화 적용


# %% [markdown]
# 아래 그림은 정보가 이 계층을 통해 어떻게 흐르는지 보여줍니다. 열은 컨텍스트 시퀀스에 대한 가중치 합계를 나타냅니다.
#
# 단순화를 위해 잔차 연결은 표시되지 않습니다.

# %% [markdown]
# <table>
# <tr>
#   <th>The cross attention layer</th>
# </tr>
# <tr>
#   <td>
#    <img width=430 src="https://www.tensorflow.org/images/tutorials/transformer/CrossAttention-new-full.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 출력 길이는 'query' 시퀀스의 길이이며, 컨텍스트 'key/value' 시퀀스의 길이는 아닙니다.
#
# 아래 다이어그램은 더욱 단순화되었습니다. 전체 "attention score" 행렬을 그릴 필요는 없습니다.
# 요점은 각 'query' 위치가 컨텍스트의 모든 'key/value' 쌍을 볼 수 있지만 query 간에 정보가 교환되지는 않는다는 것입니다.

# %% [markdown]
# <table>
# <tr>
#   <th>Each query sees the whole context.</th>
# </tr>
# <tr>
#   <td>
#    <img width=430 src="https://www.tensorflow.org/images/tutorials/transformer/CrossAttention-new.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# sample input을 시험삼아 수행해 봅니다.

# %%
# CrossAttention 레이어 예시 생성
# 포르투갈어와 영어 임베딩의 형상 출력
# CrossAttention 레이어를 영어 및 포르투갈어 임베딩에 적용하고 결과 형상 출력

# %% [markdown]
# ### global self attention layer

# %% [markdown]
# 이 계층은 컨텍스트 시퀀스를 처리하고 해당 길이에 따라 정보를 전파하는 역할을 담당합니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The global self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/SelfAttention.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# GlobalSelfAttention 클래스는 BaseAttention을 상속받아 구현됩니다.  
# call 메소드는 레이어의 입력(x)을 받아 처리합니다. 여기서 self.mha는 다중 헤드 어텐션 레이어로, 쿼리(query), 키(key), 값(value) 모두 동일한 입력 x를 사용합니다. 이는 셀프 어텐션 메커니즘을 구현하는 것으로, 입력 시퀀스 내의 각 위치가 서로의 정보를 참조합니다.

# %%
class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    # 셀프 어텐션: 쿼리, 키, 값 모두 동일한 입력 x를 사용
    # 어텐션 결과와 원래 입력을 더함 (잔차 연결)
    # 레이어 정규화 적용


# %%
# GlobalSelfAttention 레이어 예시 생성
# num_heads=2, key_dim=512인 샘플 글로벌 셀프 어텐션 레이어(sample_gsa) 생성
# 포르투갈어 임베딩의 형상 출력
# GlobalSelfAttention 레이어를 포르투갈어 임베딩에 적용하고 결과 형상 출력

# %% [markdown]
# 다음과 같이 그릴 수 있습니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The global self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img width=330 src="https://www.tensorflow.org/images/tutorials/transformer/SelfAttention-new-full.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 잔여 연결은 생략되었습니다.
# 다음과 같이 그리는 것이 더 간결하고 정확합니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The global self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img width=500 src="https://www.tensorflow.org/images/tutorials/transformer/SelfAttention-new.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# ### causal self attention layer

# %% [markdown]
# 이 레이어는 출력 시퀀스에 대해 global self attention 레이어와 유사한 작업을 수행합니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The causal self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/CausalSelfAttention.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 이는 인코더의 전역 self attention 레이어와 다르게 처리되어야 합니다.

# %% [markdown]
# CausalSelfAttention 클래스는 BaseAttention을 상속받아 구현됩니다. call 메소드는 레이어의 입력(x)을 받아 처리합니다. 여기서 self.mha는 다중 헤드 어텐션 레이어로, 쿼리(query), 키(key), 값(value) 모두 동일한 입력 x를 사용합니다. use_causal_mask=True는 인과적 마스크를 적용하여, 시퀀스의 각 위치가 그 이전 위치들의 정보만 참조하도록 합니다.

# %%
class CausalSelfAttention(BaseAttention):
  def call(self, x):
    # 인과적 셀프 어텐션: 쿼리, 키, 값 모두 동일한 입력 x를 사용하며 인과적 마스크 적용
    # 어텐션 결과와 원래 입력을 더함 (잔차 연결)
    # 레이어 정규화 적용


# %% [markdown]
# 인과 마스크는 각 위치가 이전 위치만 액세스할 수 있도록 보장합니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The causal self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img width=330 src="https://www.tensorflow.org/images/tutorials/transformer/CausalSelfAttention-new-full.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 이 레이어를 보다 간결하게 표현하면 다음과 같습니다.

# %% [markdown]
# <table>
# </tr>
#   <th colspan=1>The causal self attention layer</th>
# <tr>
# <tr>
#   <td>
#    <img width=430 src="https://www.tensorflow.org/images/tutorials/transformer/CausalSelfAttention-new.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# layer 테스트:
#
# CausalSelfAttention 클래스를 사용하여 num_heads=2, key_dim=512인 샘플 인과적 셀프 어텐션 레이어(sample_csa)를 생성합니다. en_emb는 영어 임베딩을 나타내며, 이의 형상을 출력하여 확인합니다. 그 후, sample_csa(en_emb)는 영어 임베딩에 인과적 셀프 어텐션 레이어를 적용합니다.

# %%
# CausalSelfAttention 레이어 예시 생성
# 영어 임베딩의 형상 출력
# CausalSelfAttention 레이어를 영어 임베딩에 적용하고 결과 형상 출력

# %% [markdown]
# 초기 시퀀스 요소의 출력은 이후 요소에 의존하지 않으므로 레이어 적용 전 또는 후에 요소를 자르는지는 중요하지 않습니다.

# %%
# 첫 번째 방법: 입력 시퀀스의 첫 3개 토큰에 대해서만 인과적 셀프 어텐션 적용
# 두 번째 방법: 전체 입력 시퀀스에 인과적 셀프 어텐션을 적용하고 첫 3개 토큰의 결과만 추출
# 두 결과 간의 최대 차이 계산

# %% [markdown]
# ### feed forward network

# %% [markdown]
# 또한 transformer에는 인코더와 디코더 모두에 이 point-wise feed-forward network가 포함되어 있습니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The feed forward network</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/FeedForward.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 네트워크는 중간에 ReLU 활성화가 있는 두 개의 선형 레이어(`tf.keras.layers.Dense`)와 드롭아웃 레이어로 구성됩니다. Attention 레이어와 마찬가지로 여기 코드에도 잔차 연결 및 정규화도 포함됩니다.
#
# FeedForward 클래스는 Keras의 Layer 클래스를 상속받아 정의됩니다. 생성자(__init__)에서는 모델의 차원(d_model), 피드포워드 네트워크의 내부 차원(dff), 그리고 드롭아웃 비율(dropout_rate)을 받아 내부 레이어를 초기화합니다. 내부에는 두 개의 Dense 레이어와 드롭아웃 레이어가 순차적으로 구성됩니다.
#
# call 메소드는 실제로 이 레이어가 입력 데이터 x에 적용되는 방식을 정의합니다. 여기서는 순차적 레이어 self.seq를 입력에 적용하고, 그 결과를 원래의 입력 x와 더하는 잔차 연결(residual connection)을 수행합니다. 그 후 레이어 정규화(self.layer_norm)를 적용하여 출력합니다.

# %%
class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    # 순차적 레이어 정의: Dense -> Dense -> Dropout
  def call(self, x):
    # 입력에 순차적 레이어 적용 후 원래 입력과 덧셈 (잔차 연결)


# %% [markdown]
# 레이어를 테스트하면 출력은 입력과 모양이 동일합니다.

# %%
# FeedForward 레이어 예시 생성
# 영어 임베딩의 형상 출력
# FeedForward 레이어를 영어 임베딩에 적용하고 결과 형상 출력

# %% [markdown]
# ### encoder layer

# %% [markdown]
# 인코더에는 'N' 인코더 레이어 스택이 포함되어 있습니다. 각 `EncoderLayer`에는 `GlobalSelfAttention` 및 `FeedForward` 레이어가 포함되어 있습니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The encoder layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/EncoderLayer.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 'EncoderLayer'의 정의는 다음과 같습니다.

# %%
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1):
    # 글로벌 셀프 어텐션 레이어 초기화
    # 피드포워드 네트워크 레이어 초기화
  def call(self, x):
    # 글로벌 셀프 어텐션 적용
    # 피드포워드 네트워크 적용


# %% [markdown]
# 출력은 입력과 동일한 모양을 갖게 됩니다.

# %%
# EncoderLayer 레이어 예시 생성
# 포르투갈어 임베딩의 형상 출력
# EncoderLayer 레이어를 포르투갈어 임베딩에 적용하고 결과 형상 출력

# %% [markdown]
# ### encoder

# %% [markdown]
# 다음으로 인코더를 빌드합니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The encoder</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/Encoder.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 인코더는 다음으로 구성됩니다.
#
# - 입력의 'PositionalEmbedding' 레이어.
# - 'EncoderLayer' 레이어 스택.
#
# Encoder 클래스는 Keras의 Layer 클래스를 상속받아 정의됩니다. 생성자(__init__)에서는 모델의 차원(d_model), 인코더 레이어 수(num_layers), 어텐션 헤드의 수(num_heads), 피드포워드 네트워크 차원(dff), 어휘 사전 크기(vocab_size), 드롭아웃 비율(dropout_rate)을 받아 내부 레이어를 초기화합니다.  
#
# call 메소드는 실제로 이 레이어가 입력 데이터 x에 적용되는 방식을 정의합니다. 먼저, 위치 임베딩 레이어를 적용한 후, 드롭아웃을 적용합니다. 그 후 각 인코더 레이어를 순차적으로 적용합니다.

# %%
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads,
    # 위치 임베딩 레이어 초기화
    # 인코더 레이어들을 리스트로 초기화
  def call(self, x):
    # 입력 x는 토큰 ID (배치 크기, 시퀀스 길이)
    # 드롭아웃 적용
    # 각 인코더 레이어를 순차적으로 적용


# %% [markdown]
# encoder 테스트:
#
# 4개의 레이어, 512차원의 모델, 8개의 어텐션 헤드, 2048차원의 피드포워드 네트워크, 그리고 8500 크기의 어휘 사전을 가진 샘플 인코더(sample_encoder)를 생성

# %%
# 인코더 인스턴스 생성
# 인코더에 포르투갈어 입력 적용
# 입력 및 출력 형상 출력

# %% [markdown]
# ### decoder layer

# %% [markdown]
# 디코더의 스택은 약간 더 복잡하며 각 `DecoderLayer`에는 `CausalSelfAttention`, `CrossAttention` 및 `FeedForward` 레이어가 포함되어 있습니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The decoder layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/DecoderLayer.png"/>
#   </td>
# </tr>
# </table>

# %%
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
    # 인과적 셀프 어텐션 레이어 초기화
    # 교차 어텐션 레이어 초기화
    # 피드포워드 네트워크 레이어 초기화
  def call(self, x, context):
    # 인과적 셀프 어텐션 적용
    # 교차 어텐션 적용
    # 마지막 어텐션 점수를 시각화를 위해 저장
    # 피드포워드 네트워크 적용


# %% [markdown]
# decoder layer 테스트:

# %%
# DecoderLayer 레이어 예시 생성
# 디코더 레이어에 영어 임베딩과 포르투갈어 임베딩 적용
# 입력 및 출력 형상 출력

# %% [markdown]
# ### decoder

# %% [markdown]
# `Encoder`와 유사하게 `Decoder`는 `PositionalEmbedding`과 `DecoderLayer` 스택으로 구성됩니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The embedding and positional encoding layer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/Decoder.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# `tf.keras.layers.Layer`를 확장하여 디코더를 정의합니다.

# %%
class Decoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size,
    # 위치 임베딩 레이어 초기화
  def call(self, x, context):
    # 입력 x는 토큰 ID (배치 크기, 타겟 시퀀스 길이)
    # 각 디코더 레이어 순차적 적용
    # 마지막 레이어의 어텐션 점수 저장
    # 최종 출력 형상: (배치 크기, 타겟 시퀀스 길이, d_model)


# %% [markdown]
# decoder 테스트:

# %%
# 디코더 인스턴스 생성
# 디코더에 영어 입력 데이터와 포르투갈어 임베딩 적용
# 입력 및 출력 형상 출력

# %%
# 디코더의 마지막 어텐션 점수의 형상 확인

# %% [markdown]
# Transformer 인코더와 디코더를 만들었으면 이제 Transformer 모델을 구축하고 훈련할 차례입니다.

# %% [markdown]
# ## Transformer

# %% [markdown]
# 이제 'Encoder'와 'Decoder'가 생겼습니다. 'Transformer' 모델을 완성하려면 이들을 함께 모으고 각 위치의 결과 벡터를 출력 토큰 확률로 변환하는 최종 선형('Dense') 레이어를 추가해야 합니다.
#
# 디코더의 출력은 이 최종 선형 레이어의 입력입니다.

# %% [markdown]
# <table>
# <tr>
#   <th colspan=1>The transformer</th>
# <tr>
# <tr>
#   <td>
#    <img src="https://www.tensorflow.org/images/tutorials/transformer/transformer.png"/>
#   </td>
# </tr>
# </table>

# %% [markdown]
# 인코더와 디코더는 각각 입력 어휘 사전 크기(input_vocab_size), 타겟 어휘 사전 크기(target_vocab_size), 레이어 수(num_layers), 모델 차원(d_model), 어텐션 헤드 수(num_heads), 피드포워드 네트워크 차원(dff), 드롭아웃 비율(dropout_rate)을 기반으로 설정됩니다.  
#
# call 메소드는 모델의 입력 데이터 inputs를 받아 처리합니다. 입력은 컨텍스트(context)와 타겟 시퀀스(x)로 구성됩니다. 먼저 인코더를 통해 컨텍스트를 처리하고, 그 결과를 디코더에 전달하여 타겟 시퀀스를 처리합니다. 최종적으로, Dense 레이어(self.final_layer)를 통해 최종 출력(logits)을 계산합니다.

# %%
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, num_heads, dff,
    # 인코더 초기화
    # 디코더 초기화
    # 최종 출력을 위한 Dense 레이어
  def call(self, inputs):
    # 입력: 컨텍스트와 타겟 시퀀스
    # 인코더를 통한 컨텍스트 처리
    # 디코더를 통한 타겟 시퀀스 처리
    # 최종 출력 계산
    # 필요 시 케라스 마스크 제거
    # 최종 출력 반환


# %% [markdown]
# ### Hyperparameters

# %% [markdown]
# 이 예제를 작고 상대적으로 빠르게 유지하기 위해 레이어 수(`num_layers`), 임베딩의 차원(`d_model`), `FeedForward` 레이어의 내부 차원(`dff`)을 줄입니다. 원본 Transformer 문서에 설명된 기본 모델은 `num_layers=6`, `d_model=512` 및 `dff=2048`을 사용했습니다.
#
# self-attention 헤드의 수는 동일하게 유지됩니다(`num_heads=8`).

# %%
# 원본 하이퍼파라미터
# num_layers = 4
# d_model = 128
# dff = 512
# num_heads = 8
# dropout_rate = 0.1
# ✅ 학습이 실제로 이루어지도록 개선된 버전 (30분 정도 소요)

# %% [markdown]
# 'Transformer' 모델을 인스턴스화합니다.
#
#

# %%
# 트랜스포머 모델 인스턴스 생성

# %% [markdown]
# Test

# %%
# 트랜스포머 모델에 포르투갈어와 영어 입력 적용
# 입력 및 출력 형상 출력

# %%
# 트랜스포머 모델의 디코더에서 마지막 디코더 레이어의 어텐션 점수 확인

# %% [markdown]
# 모델의 요약을 출력:

# %%
# 변수 정보

# %% [markdown]
# ## Training

# %% [markdown]
# ### optimizer 설정

# %% [markdown]
# 원래 Transformer [논문](https://arxiv.org/abs/1706.03762)의 공식에 따라 사용자 정의 학습률 스케줄러와 함께 Adam 최적화 프로그램을 사용합니다.
#
# $$\Large{lrate = d_{model}^{-0.5} * \min(step{\_}num^{-0.5}, step{\_}num \cdot warmup{\_}steps^{-1.5})}$$
#
# 초기에는 학습률을 서서히 증가시키는 "웜업(warmup)" 기간을 거치며, 이후 학습률을 점차 감소시킵니다.

# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=2000):  # 4000 → 2000 (더 빠른 학습 시작)
  def __call__(self, step):


# %% [markdown]
# 최적화 프로그램을 인스턴스화합니다(이 예에서는 `tf.keras.optimizers.Adam`입니다).

# %%
# 맞춤형 학습률 스케줄 생성
# Adam 옵티마이저 초기화

# %% [markdown]
# 사용자 정의 learning rate scheduler Test

# %%
# 40000 스텝까지의 학습률 스케줄 시각화

# %% [markdown]
# ### loss and metric 설정

# %% [markdown]
# 대상 시퀀스는 패딩되어 있으므로 손실을 계산할 때 패딩 마스크를 적용하는 것이 중요합니다. 교차 엔트로피 손실 함수(`tf.keras.losses.SparseCategoricalCrossentropy`)를 사용합니다.
#
# masked_loss 함수는 레이블이 0인 부분(일반적으로 패딩 부분)을 마스크 처리하여 이 부분의 손실을 계산에서 제외합니다.

# %%
def masked_loss(label, pred):
  # SparseCategoricalCrossentropy 손실 함수 초기화
  # 손실 계산
  # 마스크를 손실의 데이터 타입으로 캐스팅
  # 마스크를 적용한 손실의 합을 마스크의 합으로 나누어 평균 손실 계산
def masked_accuracy(label, pred):
  # 예측값에서 가장 큰 값의 인덱스를 선택
  # 레이블을 예측값의 데이터 타입으로 캐스팅
  # 예측값과 레이블이 일치하는지 여부
  # 마스크 적용하여 정확한 예측만 선택
  # 일치 여부와 마스크를 float 타입으로 캐스팅
  # 마스크를 적용한 정확도의 합을 마스크의 합으로 나누어 평균 정확도 계산


# %% [markdown]
# ### model Train

# %% [markdown]
# 모든 구성 요소가 준비되면 `model.compile`을 사용하여 학습 절차를 구성한 다음 `model.fit`을 사용하여 실행합니다.

# %%
# 트랜스포머 모델 컴파일

# %%
# 시작 시간 기록
# 학습 품질 향상을 위한 콜백 설정
# 모델 훈련
# 실행 시간 계산 및 출력

# %% [markdown]
# ## 추론 실행

# %% [markdown]
# 이제 변환을 수행하여 모델을 테스트할 수 있습니다. 추론에는 다음 단계가 사용됩니다.
#
# * 포르투갈어 토크나이저(`tokenizers.pt`)를 사용하여 입력 문장을 인코딩합니다. 이것은 인코더 입력입니다.
# * 디코더 입력은 `[START]` 토큰으로 초기화됩니다.
# * 패딩 마스크와 미리 보기 마스크를 계산합니다.
# * 그런 다음 `디코더`는 `인코더 출력`과 자체 출력(self-attention)을 보고 예측을 출력합니다.
# * 예측된 토큰을 디코더 입력에 연결하고 디코더에 전달합니다.
# * 이 접근 방식에서 디코더는 예측한 이전 토큰을 기반으로 다음 토큰을 예측합니다.

# %%
class Translator(tf.Module):
  def __init__(self, tokenizers, transformer):
  def __call__(self, sentence, max_length=MAX_TOKENS):
    # 입력 문장이 포르투갈어이므로 `[START]`와 `[END]` 토큰을 추가
    # 출력 언어가 영어이므로 영어 `[START]` 토큰으로 초기화
    # 동적 루프 추적을 위해 `tf.TensorArray` 사용
      # `seq_len` 차원에서 마지막 토큰 선택
      # `predicted_id`를 출력에 연결
    # 루프의 마지막 반복에서 계산된 어텐션 가중치를 사용할 수 없으므로,
    # 루프 밖에서 다시 계산


# %% [markdown]
# 'Translator' 클래스의 인스턴스를 만들고 몇 번 시도해 봅니다.

# %%

# %%
#함수는 세 가지 인자를 받습니다:
# sentence: 번역할 원문 문장
# tokens: 토큰화된 번역 결과입니다. 이 값은 numpy 배열로 변환되고 UTF-8로 디코딩됩니다.
# ground_truth: 실제 번역 결과
def print_translation(sentence, tokens, ground_truth):
  # 입력 문장 출력
  # 모델의 예측 번역 출력
  # 실제 번역 (ground truth) 출력


# %% [markdown]
# Example 1:

# %%
# 포르투갈어 문장
# 실제 번역 (ground truth)
# Translator를 사용하여 포르투갈어 문장 번역
# 번역 결과 출력

# %% [markdown]
# Example 2:

# %%

# %% [markdown]
# Example 3:

# %%

# %% [markdown]
# ## attention plot 생성

# %% [markdown]
# 이전 섹션에서 생성한 'Translator' 클래스는 모델의 내부 작업을 시각화하는 데 사용할 수 있는 attention 히트맵 dictionary를 반환합니다.
#
# 예를 들어:

# %%

# %% [markdown]
# 토큰이 생성될 때 attention을 plot하는 함수를 만듭니다.

# %%
def plot_attention_head(in_tokens, translated_tokens, attention):
  # 모델이 출력에서 `<START>`를 생성하지 않았으므로 이를 생략합니다.
  # 입력 토큰 레이블
  # 번역된 토큰 레이블


# %%
# 어텐션 가중치의 형상: `(batch=1, num_heads, seq_len_q, seq_len_k)`

# %% [markdown]
# 입력(포르투갈어) 토큰은 다음과 같습니다.

# %%
# 입력 문장을 텐서로 변환
# 포르투갈어 토크나이저를 사용하여 토큰화
# 토큰 인덱스를 실제 단어로 변환

# %% [markdown]
# 다음은 출력(영어 번역) 토큰입니다.

# %%

# %%
# 어텐션 맵 시각화

# %%
#번역할 문장(sentence), 번역된 토큰(translated_tokens), 어텐션 헤드들의 가중치(attention_heads)를 입력으로 받습니다.
def plot_attention_weights(sentence, translated_tokens, attention_heads):
  # 입력 문장을 텐서로 변환하고 토큰화
  # 시각화를 위한 그래프 설정
  # 각 어텐션 헤드에 대한 어텐션 맵을 그래프로 표시

# %%
# 주어진 문장, 번역된 토큰, 첫 번째 어텐션 헤드의 가중치를 사용하여 어텐션 맵 시각화

# %%
