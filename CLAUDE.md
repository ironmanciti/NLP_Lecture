# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 저장소 성격

한국어 NLP 강의용 Jupyter notebook 모음입니다. 코드는 라이브러리가 아니라 **수업 자료**이며, 셀별로 마크다운 설명 + Python 코드 + 실행 출력이 학습 흐름을 따라 배열되어 있습니다. notebook을 수정·추가할 때 이 교육적 흐름을 깨지 않는 것이 가장 중요합니다.

대상 환경은 대부분 **Google Colab (GPU)** 입니다. 로컬 Windows에서는 GPU/메모리/패키지 차이로 일부 셀이 실행되지 않을 수 있습니다 — 수정 후 "실행해서 통과시킨다"는 가정으로 검증을 강요하지 마세요. 사용자가 Colab에서 직접 돌립니다.

## 커리큘럼 구조

번호 prefix가 학습 순서이자 주제 단계입니다. 셀의 마크다운 헤더는 한국어로 일관되게 쓰여 있고, 상호 참조(예: 140 의 BERT 감성분석을 141 에서 NER 로 확장, 300 ↔ 301 의 동일 실습 비교)를 가정한 경우가 있어 임의로 번호를 재배치하지 마세요.

| 번호 | 주제 | 핵심 기술 스택 |
| ---- | ---- | -------------- |
| 010 | Tokenizer 비교 (Keras / KoNLPy Okt / SentencePiece / tiktoken) | tensorflow.keras, konlpy, sentencepiece, tiktoken |
| 020 | TF-IDF vs Sentence Embedding 유사도 | sklearn, sentence-transformers (`nlpai-lab/KURE-v1`) |
| 030 | 수강생 실습 — 토큰화 → 임베딩 → 유사도 (KURE-v1 단일 모델) | sentence-transformers `nlpai-lab/KURE-v1` |
| 130 | Transformer 구조 처음부터 구현 | tf.keras |
| 140 | BERT fine-tuning — Naver 영화 리뷰 감성분석 (sequence classification) | `bert-base-multilingual-cased`, Trainer (PyTorch) |
| 141 | BERT fine-tuning — KLUE-NER 개체명 인식 (token classification) | `klue/bert-base`, datasets, evaluate/seqeval, Trainer |
| 150 | Autoregressive 생성 (HyperCLOVAX-0.5B) | transformers, torch |
| 200 | PEFT LoRA fine-tuning — KorQuAD 한국어 QA | `Qwen/Qwen2.5-0.5B`, peft, datasets, Trainer |
| 300 | 실습: 5가지 prompt 기법 비교 — 로컬 HyperCLOVAX | transformers, `HyperCLOVAX-SEED-Text-Instruct-1.5B`, pydantic |
| 301 | 실습: 5가지 prompt 기법 비교 — Gemini API (300 의 Gemini 판) | google.genai (`gemini-2.5-flash-lite`), pydantic |
| 500 | HuggingFace `pipeline` 빠른 둘러보기 | transformers (다양한 사전학습 모델) |

`Template_Creation.ipynb`는 별개의 유틸리티입니다: 현재 디렉토리의 모든 `.ipynb`에서 **마크다운 셀과 코드의 주석/`def`/`class` 줄만 남긴** 빈 학습용 템플릿(`template_*.ipynb`)을 생성합니다. 강의 배포용이므로 코드 노트북을 의미 있게 바꾼 뒤에는 사용자가 이 셀을 다시 실행해 템플릿을 재생성할 수 있다는 점을 인지하세요.

옛 번호 체계(`015_word2vec.ipynb`, `030_IMDB_movie_reviews.ipynb`, `080_language_translation_*.ipynb` 등)와 `template_*.ipynb` 는 **삭제**된 상태입니다. 또한 번호 재정비가 있었습니다 — `320→140`, `400→200`, `300→500`, `600→301`, `601→300`. 위 표의 번호가 현재 기준이며, 옛 번호로 참조된 코드를 복원하지 마세요.

## 외부 데이터 / 모델 의존성

notebook들이 인터넷에서 가져오는 자원이 많습니다. 새로 추가하지 말고, 기존 패턴을 따르세요:

- **이 저자의 GitHub 데이터** — `https://github.com/ironmanciti/infran_NLP/raw/main/data/...` 및 `https://raw.githubusercontent.com/ironmanciti/infran_NLP/...` 에 NSMC 등이 있고 `tf.keras.utils.get_file()`로 캐시 다운로드.
- **KorQuAD / KLUE 데이터셋** — 200은 `KorQuAD/squad_kor_v1`, 141은 `klue` (`ner`) 를 HuggingFace `datasets`로 받습니다.
- **HuggingFace 모델** — `naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B` (150, 500), `…-1.5B` (300, 토큰 로그인 필요), `bert-base-multilingual-cased` (140), `klue/bert-base` (141), `Qwen/Qwen2.5-0.5B` (200), `nlpai-lab/KURE-v1` (020·030). 모델 ID는 그대로 유지하세요.
- **API 키 / 토큰** — 301은 `.env`의 `GOOGLE_API_KEY`를, 300은 `.env`의 `HF_TOKEN`을 `python-dotenv`로 로드합니다. `.env`는 `.gitignore`에 포함되어 있고 절대 커밋되면 안 됩니다. 301의 모델은 `gemini-2.5-flash-lite`로 고정되어 있습니다.

## 실행 방식

- 강의 진행 시: 셀 위에서부터 순서대로 실행하는 선형 흐름이 전제입니다. 셀 순서를 바꾸는 리팩터링은 학습 흐름을 깨므로 피하세요.
- 로컬에서 빠른 검증: `Template_Creation.ipynb`처럼 GPU·외부 다운로드가 없는 셀은 로컬에서 돌릴 수 있습니다. 그 외(130, 140, 141, 150, 200, 300, 500 등)는 Colab GPU 전제입니다. 단 301(Gemini)은 API 호출이라 GPU가 필요 없습니다.
- 단일 셀 실행/검증을 자동화하려면 `jupyter nbconvert --to notebook --execute <file>.ipynb`이지만, 위의 외부 의존성 때문에 CI식 일괄 실행은 권장하지 않습니다.

## Jupytext 페어링

모든 notebook은 [jupytext.toml](jupytext.toml) 설정에 따라 동명의 `.py` (percent format)와 짝지어져 있습니다. 예: [010_Tokenizers.ipynb](010_Tokenizers.ipynb) ↔ [010_Tokenizers.py](010_Tokenizers.py).

- **편집 후 동기화**: 어느 쪽이든 한쪽을 수정했으면 `jupytext --sync <name>.ipynb`. 전체 일괄: `jupytext --sync *.ipynb`.
- **pre-commit hook**: [.pre-commit-config.yaml](.pre-commit-config.yaml)에 hook이 설정되어 있어, `pre-commit install` 후에는 커밋 시 자동으로 `--sync`가 돌아갑니다. 최초 1회만 `pip install pre-commit && pre-commit install`.
- **셀 출력은 .ipynb 에만**: jupytext.toml의 `cell_metadata_filter = "-all"` 때문에 .py 에는 코드와 마크다운만 남고 출력은 .ipynb 에만 보존됩니다. 강의용 출력 보존 원칙과 일치.
- **둘 다 git 추적**: .ipynb 와 .py 모두 커밋합니다. 코드 리뷰/diff는 .py 쪽이 훨씬 가벼우므로 PR 검토 시 .py 를 보세요.
- **충돌 시**: 두 파일이 따로 수정돼 어긋난 경우 jupytext가 더 새로 수정된 쪽을 기준으로 다른 쪽을 덮어씁니다. 의심스러우면 `jupytext --diff <name>.ipynb`로 먼저 확인.
- **새 notebook 추가 시**: `jupytext --set-formats ipynb,py:percent <new>.ipynb` 한 번 실행하면 그 이후 sync 가능. (`jupytext.toml`의 `formats`가 전역으로 적용되지만, 메타데이터를 한 번 기록해 두는 게 안전합니다.)

## notebook 편집 시 주의사항

- 코드 셀의 한국어 주석은 학습 자료의 핵심입니다. 일반 코드베이스의 "주석 줄여라" 규칙을 이 저장소에는 **적용하지 마세요** — 셀의 한국어 설명 주석은 의도된 것이며 보존·보강 대상입니다.
- 출력(output) 셀은 강의 시 학생이 결과를 미리 볼 수 있도록 의도적으로 커밋되어 있습니다. 출력을 비우거나 재실행으로 덮어쓰는 작업은 사용자가 명시적으로 요청할 때만 하세요.
- 코드/주석/문자열 안에서 한자(漢字, CJK Unified Ideographs)는 사용하지 마세요. 한국어가 필요하면 한글만. (전역 규칙)
- notebook은 JSON이지만 큰 파일이 많아 (130_Transformer ~700 KB, 500_HuggingFace_QuickStart ~550 KB) `Read` 시 `offset`/`limit`로 부분만 읽는 게 안전합니다.
- 출력에서 큰 텐서/이미지/HTML이 나오면 도구 출력에 `Outputs are too large to include. Use Bash with: cat <notebook_path> | jq '.cells[N].outputs'` 안내가 뜹니다. 필요할 때 그 명령으로 추출하세요.
