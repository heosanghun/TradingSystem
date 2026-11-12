# 다중 모달 국면 적응형 강화학습과 이중 프로세스 지식 증류 기반 트레이딩 시스템

본 저장소는 학위논문 *시장 국면 적응형 다중 모달 앙상블 강화학습과 이중 프로세스 지식 증류 기반의 지능형 트레이딩 시스템*의 실험·평가를 재현하기 위한 파이썬 구현체이다. 사후 결과 보정 계층을 두지 않으며, 봉 $t$ 종가 시점에 산출된 목표 비중이 봉 $t+1$ 수익률에 집행되는 시점 인과적 포트폴리오 회계(`CausalPortfolioBacktester.run_decisions`)와 출처 대장(`ledger/numbers.csv`)을 통해 모든 지표를 기록한다.

---

## 1. 개요

전체 파이프라인은 오프라인 교사(System 2)와 온라인 학생(System 1)의 이중 프로세스 구조로 구성된다.

- **연구 1 (`src/ts/study1/`)**: 60봉 캔들·거래량 차트 이미지(`ResNet-18 + CBAM`), 16개 기술 지표, 뉴스 감성 지수가중 이동평균 요약을 결합한 상태 표현 위에서 퍼징된 추세 스캐닝(Trend Scanning) 라벨과 전환 임계 게이트(`theta = 0.60`) 기반 `XGBoost` 국면 분류기를 학습하고, 상승·하락·횡보 3국면별 `PPO + PER` 전문가 풀을 최근 720봉 반사실적 샤프 비율 기반 볼츠만 가중으로 결합한다.
- **연구 2 (`src/ts/study2/`)**: 다중 자산 시계열에 대해 그레인저 인과 검정과 전이 엔트로피 추정을 수행하여 일별 스펙트럴 하이퍼그래프를 구성하고, 인용 정합성 검증기(`CitationVerifier`)가 결합된 알파 분석가·리스크 관리자·메타 포트폴리오 매니저 토론을 통해 최종 교사 비중 $w_t^*$와 합의 논거 텍스트 $C_t$를 산출한다.
- **연구 3 (`src/ts/study3/`)**: 교사 비중을 가우시안 소프트 타깃 분포($\sigma_w = 0.25, \tau = 2.0$)로 변환하고 합의 논거의 `FinBERT` `[CLS]` 임베딩($768$차원)을 추출한 뒤, 삼중 손실($L_{\text{task}} + L_{\text{policy}} + L_{\text{reasoning}}$)로 경량 `Student1DCNN`을 오프라인 증류한다. 온라인 추론(`Student1DCNN.act_online`)에서는 1D-CNN 백본과 액터 헤드만 실행하며 교사 모델·LLM·사영 헤드는 호출하지 않는다.

---

## 2. 디렉토리 구성

```text
TradingSystem/
├── configs/
│   └── appendix_a_config.json       # 부록 A (표 A.1~A.4) 및 제VI장 실험 설정 (JSON)
├── data/
│   ├── DATA_SOURCES.md              # 입력 데이터 출처, 시점 가용성 및 전처리 기준 문서
│   └── collect_inputs.py            # 입력 시계열 수집 및 무결성 점검 도구
├── src/ts/
│   ├── config.py                    # 불변 설정 로더 및 설정 파일 SHA-256 해시 계산
│   ├── ledger.py                    # 본 실행 검증 게이트(`RunContext`) 및 대장 기록기
│   ├── data/
│   │   ├── temporal_guard.py        # 미래 시점 참조 및 경계 누수 런타임 검사기
│   │   ├── binance_fetcher.py       # 바이낸스 현물 1시간·1분 OHLCV 수집 및 결측 봉 집계
│   │   ├── news_fetcher.py          # UTC 발행 시각 기준 뉴스 중복 제거 및 24시간 EWMA 요약
│   │   ├── macro_fetcher.py         # 다중 암호화폐·거시 자산 수익률 정렬 및 결측 점검
│   │   ├── lob_processor.py         # 100구간 × 40특징 미시구조 윈도우 추출 및 정규화
│   │   └── tardis_adapter.py        # L2 호가 스냅샷 및 체결 스트림 40특징 변환기
│   ├── study1/
│   │   ├── chart_renderer.py        # 60봉 캔들·거래량 uint8 차트 렌더러 및 배치 지연 정규화
│   │   ├── encoders.py              # 다중 모달 인코더, 2토큰 교차 어텐션 및 모달 소거 분기
│   │   ├── regime_labeling.py       # Phase 1 SMA-50 라벨러 및 Phase 2 퍼징 추세 스캐닝 라벨러
│   │   ├── regime_classifier.py     # XGBoost 국면 분류기 및 신뢰도 전환 임계 게이트
│   │   ├── per_buffer.py            # 온폴리시 롤아웃 내 세그먼트 트리 우선순위 경험 재생
│   │   ├── ppo_expert.py            # 국면별 보상 함수 기반 PPO 전문가 및 수렴 로그 기록
│   │   ├── ensemble.py              # 롤링 샤프 볼츠만 동적 가중 앙상블
│   │   └── attribution.py           # 시각·기술·감성 3개 모달 블록 단위 Shapley 기여도 산출
│   ├── study2/
│   │   ├── causality.py             # 시차 그레인저 F-검정 및 3차원 KDE 조건부 전이 엔트로피 근사
│   │   ├── hypergraph.py            # 스펙트럴 군집화 하이퍼그래프 및 랜덤 워크 전이 행렬
│   │   ├── prompts.py               # 부록 B 에이전트 역할별 JSON 입출력 프롬프트
│   │   ├── llm_client.py            # 로컬 OpenAI 호환(vLLM) 및 Ollama 서빙 클라이언트와 캐시
│   │   ├── citation_verifier.py     # 엣지 ID 및 전이 엔트로피 수치 인용 정합성 검증기
│   │   └── debate.py                # 3라운드 근거 중심 토론 및 메타 포트폴리오 매니저 의사결정
│   ├── study3/
│   │   ├── teacher_signal.py        # 교사 소프트 타깃 분포, FinBERT 임베딩 및 시간 정렬
│   │   ├── student_model.py         # 부록 C 구조의 Student1DCNN 및 온라인 단독 실행 경로
│   │   └── distillation.py          # 삼중 목적 오프라인 증류 학습기 및 성질 5.1 직교성 검증
│   ├── baselines/
│   │   └── models.py                # Buy & Hold, Single PPO, 정적 앙상블(1/N), Reptile식 메타 PPO
│   ├── eval/
│   │   ├── cost_model.py            # Phase 1 고정 수수료(0.05%) 및 Phase 2 ATR+충격 동적 비용
│   │   ├── metrics.py               # 기술 통계, 단위근·자기상관 검정, 성과 지표 및 위기 구간 낙폭
│   │   ├── walk_forward.py          # 9폴드 롤링 Walk-Forward(12개월 학습, 3개월 시험) 분할기
│   │   ├── backtester.py            # 1봉 지연 집행 시점 인과 포트폴리오 백테스터
│   │   ├── significance.py          # Ledoit-Wolf HAC 검정 및 순환 블록 부트스트랩(24봉, 5,000회)
│   │   ├── shap_analysis.py         # 개별 특징 수준 KernelSHAP 보조 분석기
│   │   ├── latency_benchmark.py     # CUDA 동기화 온라인 학생 추론 지연 측정기
│   │   └── xai_metrics.py           # 인용 일치율·환각 기각률·교사-학생 행동 일치율 집계기
│   └── future_work/
│       └── film_double_buffer.py    # 부록 D 향후 연구용 FiLM 및 이중 버퍼 공유 메모리 참조 설계
├── scripts/                         # 단계별 본 실행 및 표 생성 스크립트
├── ledger/                          # 최종 수치 대장(`numbers.csv`) 및 임시 기록(`provisional.csv`)
└── tests/                           # 단위·통합·회귀 테스트 스위트
```

---

## 3. 논문과의 차이

논문 본문 서술과 현재 코드·데이터 구성 사이에 차이가 있는 항목은 다음과 같다.

1. **미시구조 입력 데이터 (제V장 3.1절, 표 5.1)**: 논문은 상위 10호가 L2 스냅샷 기반 `100틱 × 40특징` 텐서를 서술한다. `src/ts/data/tardis_adapter.py`를 통해 실제 L2 스냅샷(`book_snapshot_25`)과 체결 스트림 변환이 검증되어 있으나(2022-05-01 표본 762,983 스냅샷 기준 100틱 중앙값 10.9초), 전 기간 유료 L2 원시 자료가 없는 기본 경로(`scripts/run_study3.py`)에서는 바이낸스 1분 봉 OHLCV 및 테이커 매수 체결량으로부터 구성한 40개 미시구조 대용 특징(`lob_input_mode="subbar_1m_microstructure"`)을 사용한다. 이 대용 특징은 호가창 잔량 복원이 아니다.
2. **뉴스 자료 출처 및 건수 (제VI장 2.1절, 표 6.1)**: 논문 표 6.1은 Bloomberg·Reuters·CoinDesk API 수집 `245,000`건을 기재하고 있으나, 실제 확보된 공개 뉴스 자료(`data/news_headlines.parquet`)는 Yahoo Finance 집계 공개 데이터셋(`edaschau/bitcoin_news`, CoinDesk·Reuters 등 포함, Bloomberg 미포함)에서 `2020-08-01`~`2023-12-31` 구간 필터링과 제목 해시 중복 제거를 거친 `50,008`건(평가 구간 `34,123`건)이다.
3. **뉴스 감성 분류 모델 (제III장 1.4절, 표 A.2)**: 논문은 `DeepSeek-R1-Distill-Qwen-32B`를 이용한 3분류 감성 점수 산출을 서술하나, 현재 구현(`PointInTimeNewsStore.populate_missing_logits_with_finbert`)은 `ProsusAI/finbert`의 3분류 로짓(`negative`, `neutral`, `positive`)으로부터 식 (III.7)~(III.8)의 점수와 24시간 지수가중 이동평균(`alpha = 0.08`) 요약을 계산한다.
4. **거시·교차 자산 시간봉 범위 (제IV장 2.1절, 표 A.3)**: 논문은 암호화폐·주가지수·국채·환율·원자재 약 60개 자산의 1시간 수익률을 서술한다. 바이낸스 현물에서 전 구간 1시간 봉이 수집되는 암호화폐 20종과 달리, 미국 ETF·외환·원자재의 2020~2023년 1시간 봉은 외부 시세 API 키 또는 유료 자료가 필요하다. 일봉 시계열을 1시간 격자에 반복 기입하는 방식은 시차 24 검정에서 인위적 자기상관을 유발하므로 사용하지 않는다.
5. **전이 엔트로피 추정량 (제IV장 2.2절, 식 IV.2~IV.3)**: 24개 시차 전체 결합분포에 대한 비모수 밀도 추정 대신, 직전 1봉 수익률과 직전 2~24봉 평균 수익률로 축약한 3차원 가우시안 커널 밀도 추정(KDE) 조건부 전이 엔트로피 근사를 사용하며 후보 쌍 탐색 시 다중비교 보정은 적용하지 않는다(`src/ts/study2/causality.py`).
6. **메타 강화학습 비교 모델 (제VI장 1.3절)**: `MAMLMetaPPOBaseline`(`src/ts/baselines/models.py`)은 2차 도함수나 메타 검증 손실 역전파를 계산하는 원형 MAML이 아니라, 태스크별 내부 적응 파라미터와 초기 파라미터의 차이를 선형 보간하는 Reptile 방식 1차 메타 갱신으로 구현되어 있다.
7. **모달리티 기여도 분석 (제VI장 4.2절, 표 6.9)**: 224차원 개별 특징에 대한 표본 추출식 KernelSHAP 대신, 시각(`128`)·기술(`64`)·감성(`32`) 3개 모달 블록의 `2^3 = 8`개 연합을 전수 평가하는 블록 단위 유한 Shapley 값을 각 봉의 활성 국면 전문가 풀과 선택 행동 로짓에 대해 계산한다(`src/ts/study1/attribution.py`).
8. **소거 실험 Variant A 정의 (제III장 1.5절, 표 6.10)**: `XGBoost` 국면 분류기와 3국면 동적 앙상블 구조는 Variant D와 동일하게 유지하고 상태 인코더 입력만 기술 지표 단일 모달(기본값 `--variant-a-modality tech_only`) 또는 뉴스 감성 제거(`visual_tech_no_senti`)로 제한한다.
9. **부록 D 실시간 맥락 주입 설계**: `src/ts/future_work/film_double_buffer.py`는 향후 연구용 참조 구현으로 분리되어 있으며 제VI장 실험에는 사용되지 않는다.

---

## 4. 재현 절차

### 4.1 가상환경 구성 및 테스트

Python 3.11 이상 환경에서 로컬 가상환경을 생성하고 패키지를 설치한다.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
```

### 4.2 대장 기록 규칙 및 단계별 본 실행

`src/ts/ledger.py`의 `NumberLedger`는 실행 컨텍스트(`RunContext`)와 작업 트리의 커밋 상태를 함께 검사한다. 미커밋 변경이 있거나 축소 예산·테스트 대체 플래그·스모크 환경 변수(`TS_SMOKE_RUN=1`)가 켜진 실행은 모두 `ledger/provisional.csv`로 분리되며, 깨끗한 커밋 상태의 본 실행만 `ledger/numbers.csv`에 기록된다.

단계별 실행 순서는 다음과 같다.

```powershell
# 1. 기초 시계열 통계(표 6.1, 6.2), 단위근·자기상관 검정, Buy & Hold, 학생 온라인 추론 지연(표 6.14)
.\.venv\Scripts\python.exe scripts/run_wp0_data_and_stats.py

# 2. 연구 1: 9폴드 Walk-Forward, 표 6.9 모달 기여도, 표 6.10 소거(Variant A~D), 표 6.12 온도 T, 표 6.13 전문가 수 K
.\.venv\Scripts\python.exe scripts/run_study1.py

# 3. 연구 2: 동적 하이퍼그래프 및 근거 중심 LLM 토론 (로컬 vLLM 또는 Ollama 서버 필요)
.\.venv\Scripts\python.exe scripts/run_study2.py --llm-backend ollama --llm-endpoint http://localhost:11434 --llm-model deepseek-r1:32b

# 4. 연구 3: 오프라인 삼중 손실 추론 증류, 성질 5.1 검증, 표 6.6~6.8 제안 모델, 표 6.11 lambda_3 민감도, 표 6.15 일치율
.\.venv\Scripts\python.exe scripts/run_study3.py

# 5. 비교 모델(Single PPO, 정적 앙상블, Reptile식 메타 PPO) 및 Ledoit-Wolf 부트스트랩 유의성 검정
.\.venv\Scripts\python.exe scripts/run_baselines_and_eval.py

# 6. ledger/numbers.csv 기반 제VI장 마크다운 표 생성 및 누락 항목 점검
.\.venv\Scripts\python.exe scripts/make_tables.py
```

전 단계를 순서대로 호출하는 오케스트레이터는 `scripts/run_full_pipeline.py`이며, 빠른 배선 점검 시 `--smoke-test`를 지정하면 자동으로 `TS_SMOKE_RUN=1`이 설정되어 최종 대장을 오염시키지 않는다.

---

## 5. 하드웨어와 소요 시간

현재까지 로컬 측정 환경(Windows 11, NVIDIA GeForce RTX 5090 Laptop GPU 24GB VRAM, 시스템 RAM 31GB)에서 실측이 완료된 구간의 소요 시간과 자원 사용량은 다음과 같다. 아직 본 실행을 마치지 않은 단계의 소요 시간은 추측하여 기입하지 않으며, 각 본 실행 완료 후 실측값으로 갱신한다.

- **WP0 기초 통계 및 학생 온라인 추론 지연 (`scripts/run_wp0_data_and_stats.py`, 커밋 `6d77ce8`)**: `Student1DCNN.act_online` 단일 윈도우(`1 × 40 × 100`) 10,000회 반복 추론 지연 평균 `0.527 ms`, 중앙값 `0.462 ms`, 99백분위수 `1.462 ms`.
- **연구 1 차트 렌더링 및 폴드 1 인코딩 메모리·시간 (`T-101`, `29,932`봉 전체 기준)**: `uint8` 캔들 차트 렌더링 `8.93초`(버퍼 `4.20 GiB`), 폴드 1(`8,741`봉) 사전학습 1에폭 + 학습·시험 인코딩 포함 총 `34.71초`, 프로세스 최대 작업 집합(Peak RSS) `6.05 GiB`(`6.49 GB`), GPU 최대 할당 메모리 `1.54 GiB`.
- **연구 1 폴드 1 검증 구간 PPO 학습 예산 실측 (`T-102`, 학습 `7,283`봉·검증 `1,458`봉)**: `FinBERT` 뉴스 감성 채점(`15,883`건) `11.76초`, `ResNet-18 + CBAM` 4에폭 사전학습 및 인코딩 `60.97초`, 기본 예산 `default_rollouts_per_fold = 64`에서 3국면 × 5전문가(15개 전문가) 온폴리시 `PPO + PER` 학습 `142.06초`(`2.37분`).
- **연구 2 후보 인과 엣지 탐색 (`T-203`)**: 60종 × 1,440봉(60일 창) 기준 일별 그레인저 + KDE 전이 엔트로피 탐색 약 `4.9초`.
- **연구 1~3 및 비교 모델 9폴드 전체 본 실행**: 선행 단계 완료 후 실측 예정.
