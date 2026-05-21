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
# # RAG Agent (검색 증강 생성 에이전트)
#
# **RAG (Retrieval-Augmented Generation)** 는 외부 지식 소스에서 정보를 검색하여 LLM의 응답을 향상시키는 기술입니다.
#
# RAG 에이전트는 검색 도구를 사용하여 관련 문서를 찾고, 이를 컨텍스트로 활용하여 답변을 생성합니다.
#
# **참고**: [LangChain 공식 문서 - RAG Agent](https://docs.langchain.com/oss/python/langchain/rag)
#
# ## RAG 워크플로우
#
# 1. **인덱싱 (Indexing)**: 문서를 로드, 분할, 임베딩하여 벡터 스토어에 저장
# 2. **검색 및 생성 (Retrieval & Generation)**: 사용자 질문에 대해 관련 문서를 검색하고 답변 생성
#
# 에이전트가 필요할 때만 검색 도구를 호출합니다.
# - 장점: 유연성, 필요할 때만 검색
# - 단점: 여러 번의 모델 호출로 인한 지연 시간

# %%
# 필요한 라이브러리 설치 (Colab 등에서 최초 1회 실행)
# !pip install -q -U langchain langchain-google-genai langchain-community beautifulsoup4

# %%
from dotenv import load_dotenv
import os

load_dotenv()

# %%
from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# 모델 및 임베딩 초기화 (Gemini 사용)
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# 벡터 스토어 생성
vector_store = InMemoryVectorStore(embeddings)

# %% [markdown]
# ## 1. 인덱싱 단계
#
# 문서를 로드하고 벡터 스토어에 저장합니다.

# %% [markdown]
# ### 1.1 문서 로드 및 분할

# %%
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 웹 문서 로드
loader = WebBaseLoader("https://botpress.com/ko/blog/llm-agents")
docs = loader.load()

print(f"로드된 문서 수: {len(docs)}")
print(f"문서 길이: {len(docs[0].page_content)} 문자")

# 텍스트 분할
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

all_splits = text_splitter.split_documents(docs)
print(f"\n분할된 청크 수: {len(all_splits)}")

# %% [markdown]
# ### 1.2 벡터 스토어에 저장

# %%
# 문서를 벡터 스토어에 추가
document_ids = vector_store.add_documents(documents=all_splits)
print(f"저장된 문서 ID 수: {len(document_ids)}")

# %% [markdown]
# ### 2. 체인 기반 RAG (Chain-based RAG)
#
# 검색을 먼저 실행하고 검색 결과를 LLM에 컨텍스트로 제공

# %%
# 체인 기반 RAG 예제
from langchain_core.messages import HumanMessage

def chain_based_rag(query: str):
    """체인 기반 RAG: 항상 검색 후 답변 생성"""
    # 1. 검색
    retrieved_docs = vector_store.similarity_search(query, k=2)
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    # 2. 컨텍스트와 함께 답변 생성
    messages = [
        HumanMessage(content=f"""다음 컨텍스트를 참고하여 질문에 답변하세요:

컨텍스트:
{context}

질문: {query}

답변:""")
    ]

    response = model.invoke(messages)
    return response.content

# 체인 기반 RAG 테스트
query = "에이전트의 주요 특징은?"
print(f"질문: {query}\n")
answer = chain_based_rag(query)
print(f"답변: {answer}")

# %% [markdown]
# ### 3. 에이전트 기반 RAG (Agentic RAG)

# %% [markdown]
# ### 3.1 검색 도구 생성

# %%
from langchain.tools import tool

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """질문에 답하기 위해 관련 정보를 검색합니다.

    이 도구는 벡터 스토어에서 질문과 관련된 문서를 검색합니다.
    검색된 문서는 답변 생성에 사용됩니다.

    Args:
        query: 검색할 질문 또는 키워드
    """
    # 벡터 스토어에서 유사한 문서 검색
    retrieved_docs = vector_store.similarity_search(query, k=2)

    # 검색된 문서를 문자열로 직렬화
    serialized = "\n\n".join(
        f"출처: {doc.metadata.get('source', 'N/A')}\n내용: {doc.page_content}"
        for doc in retrieved_docs
    )

    # 문자열과 문서 객체를 함께 반환
    return serialized, retrieved_docs

# 도구 테스트
test_result = retrieve_context.invoke("LLM 에이전트의 구성 요소")
test_result

# %% [markdown]
# ### 3.2 RAG 에이전트 생성

# %%
from langchain.agents import create_agent

# 시스템 프롬프트 설정
system_prompt = (
    "당신은 블로그 게시글에서 관련 문맥(context)을 검색하는 도구에 접근할 수 있습니다. "
    "사용자의 질문에 답하기 위해 먼저 retrieve_context 도구를 사용하여 관련 정보를 검색한 후, "
    "검색된 정보를 바탕으로 정확하고 유용한 답변을 제공하세요."
)

# 에이전트 생성
agent = create_agent(
    model,
    tools=[retrieve_context],
    system_prompt=system_prompt
)

agent

# %% [markdown]
# ### 3.3 에이전트 실행

# %%
# 질문에 대한 답변 생성
query = "LLM 에이전트 프레임워크를 구성하는 핵심 구성 요소는 무엇인가요?"
# query = "에이전트의 주요 특징은 무엇인가요?"
# query = "프롬프트 엔지니어링이 중요한 이유는 무엇인가요?"

print(f"질문: {query}\n")
print("=" * 80)

# 스트리밍 방식으로 실행
for event in agent.stream(
    {"messages": [{"role": "user", "content": query}]},
    stream_mode="values",
):
    event["messages"][-1].pretty_print()

# %% [markdown]
# ## 주요 포인트 정리
#
# 1. **인덱싱**: 문서 로드 → 분할 → 임베딩 → 벡터 스토어 저장
# 2. **검색 도구**: 벡터 스토어를 래핑한 도구 생성
# 3. **에이전트 통합**: 검색 도구를 에이전트에 추가
# 4. **에이전트 기반 vs 체인 기반**: 필요에 따라 선택

# %%
