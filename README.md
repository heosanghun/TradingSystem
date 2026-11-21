# 다중 모달 국면 적응형 강화학습과 이중 프로세스 지식 증류 기반 트레이딩 시스템

본 저장소는 학위논문 *시장 국면 적응형 다중 모달 앙상블 강화학습과 이중 프로세스 지식 증류 기반의 지능형 트레이딩 시스템*의 실험·평가를 재현하기 위한 파이썬 구현체이다. 사후 결과 보정 계층을 두지 않으며, 봉 $t$ 종가 시점에 산출된 목표 비중이 봉 $t+1$ 수익률에 집행되는 시점 인과적 포트폴리오 회계(`CausalPortfolioBacktester.run_decisions`)와 출처 대장(`ledger/numbers.csv`)을 통해 모든 지표를 기록한다.

---

## 1. 개요

전체 파이프라인은 오프라인 교사(System 2)와 온라인 학생(System 1)의 이중 프로세스 구조로 구성된다.

- **연구 1 (`src/ts/study1/`)**: 60봉 캔들·거래량 차트 이미지(`ResNet-18 + CBAM`), 16개 기술 지표, 뉴스 감성 지수가중 이동평균 요약을 결합한 상태 표현 위에서 퍼징된 추세 스캐닝(Trend Scanning) 라벨과 전환 임계 게이트(`theta = 0.60`) 기반 `XGBoost` 국면 분류기를 학습하고, 상승·하락·횡보 3국면별 `PPO + PER` 전문가 풀을 최근 720봉 반사실적 샤프 비율 기반 볼츠만 가중으로 결합한다.
- **연구 2 (`src/ts/study2/`)**: 다중 자산 시계열에 대해 그레인저 인과 검정과 전이 엔트로피 추정을 수행하여 일별 스펙트럴 하이퍼그래프를 구성하고, 인용 정합성 검증기(`CitationVerifier`)가 결합된 알파 분석가·리스크 관리자·메타 포트폴리오 매니저 토론을 통해 최종 교사 비중 $w_t^*$와 합의 논거 텍스트 $C_t$를 산출한다.
- **연구 3 (`src/ts/study3/`)**: 교사 비중을 가우시안 소프트 타깃 분포($\sigma_w = 0.25, \tau = 2.0$)로 변환하고 합의 논거의 `FinBERT` `[CLS]` 임베딩($768$차원)을 추출한 뒤, 삼중 손실($L_{\text{task}} + L_{\text{policy}} + L_{\text{reasoning}}$)로 경량 `Student1DCNN`을 오프라인 증류한다. 온라인 추론(`Student1DCNN.act_online`)에서는 1D-CNN 백본과 액터 헤드만 실행하며 교사 모델·LLM·사영 헤드는 호출하지 않고, 행동 전환 신뢰도 게이트(`execute_student_weights`)를 거쳐 목표 비중을 결정한다.

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
│   │   ├── student_model.py         # 부록 C 구조의 Student1DCNN, 온라인 단독 실행 경로 및 행동 신뢰도 게이트
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
├── tools/
│   ├── score_news_sentiment_llm.py  # DeepSeek-R1 3분류 첫 토큰 로그확률 뉴스 감성 채점기
│   └── run_pipeline_after_r2.ps1    # 연구 2 완료 후 연구 3·비교 모델·표·그림 연속 실행 스크립트
├── scripts/                         # 단계별 본 실행, 내부 검증, 잠재 공간 그림 및 표 생성 스크립트
├── ledger/                          # 최종 수치 대장(`numbers.csv`) 및 임시 기록(`provisional.csv`)
└── tests/                           # 단위·통합·회귀 테스트 스위트
```

---

## 3. 논문과의 차이

논문 본문 서술과 현재 코드·데이터 구성 사이에 차이가 있는 항목은 다음과 같다.

1. **미시구조 입력 데이터 (제V장 3.1절, 표 5.1)**: 논문은 상위 10호가 L2 스냅샷 기반 `100틱 × 40특징` 텐서를 서술한다. `src/ts/data/tardis_adapter.py`를 통해 실제 L2 스냅샷(`book_snapshot_25`)과 체결 스트림 변환이 검증되어 있으나(2022-05-01 표본 762,983 스냅샷 기준 100틱 중앙값 10.9초), 전 기간 유료 L2 원시 자료가 없는 기본 경로(`scripts/run_study3.py`)에서는 바이낸스 1분 봉 OHLCV 및 테이커 매수 체결량으로부터 구성한 40개 미시구조 대용 특징(`lob_input_mode="subbar_1m_microstructure"`)을 사용한다. 이 대용 특징은 호가창 잔량 복원이 아니다.
2. **뉴스 자료 출처 및 건수 (제VI장 2.1절, 표 6.1)**: 논문 표 6.1은 Bloomberg·Reuters·CoinDesk API 수집 `245,000`건을 기재하고 있으나, 실제 확보된 공개 뉴스 자료(`data/news_headlines.parquet`)는 Yahoo Finance 집계 공개 데이터셋(`edaschau/bitcoin_news`, CoinDesk·Reuters 등 포함, Bloomberg 미포함)에서 `2020-08-01`~`2023-12-31` 구간 필터링과 제목 해시 중복 제거를 거친 `50,008`건(평가 구간 `34,123`건)이다.
3. **뉴스 감성 분류 방식 (제III장 1.4절, 표 A.2)**: `tools/score_news_sentiment_llm.py`가 로컬 Ollama 서버(`deepseek-r1:32b`, 사고 블록 사전 종료 및 `temperature = 0`)로부터 세 클래스 단어(`Negative`, `Neutral`, `Positive`)의 첫 답변 토큰 로그확률을 조회하여 `data/news_headlines.parquet`의 `logit_neg`, `logit_neu`, `logit_pos` 열에 저장하고(상위 20개 후보 밖 단어는 반환된 최소 로그확률 상한으로 대체), 이를 식 (III.7)~(III.8)의 로짓으로 사용해 24시간 지수가중 이동평균(`alpha = 0.08`) 요약을 산출한다. 사전 채점 열이 이미 존재하면 `PointInTimeNewsStore.populate_missing_logits_with_finbert` 보조 경로는 실행되지 않는다.
4. **거시·교차 자산 시간봉 범위 (제IV장 2.1절, 표 A.3)**: 논문은 암호화폐·주가지수·국채·환율·원자재 약 60개 자산의 1시간 수익률을 서술한다. 본 실행에서는 바이낸스 현물 시세(2020-08 상장 이후 전 구간 존재 암호화폐 14종 및 외환 대용 `EURUSDT`·`GBPUSDT` 2종)와 Alpaca 과거 시세로 수집한 미국 주식·채권·원자재 ETF 33종을 합쳐 실제 1시간 봉이 확보된 총 `49`개 자산을 사용한다. 직접 시간봉 원자료가 없는 지수 3종, 엔·호주달러 현물, 2020-08 시작 시점 미상장 암호화폐 6종은 제외하며, 일봉 시계열을 1시간 격자에 반복 기입하는 방식은 시차 24 검정에서 인위적 자기상관을 유발하므로 사용하지 않는다.
5. **전이 엔트로피 추정량 (제IV장 2.2절, 식 IV.2~IV.3)**: 24개 시차 전체 결합분포에 대한 비모수 밀도 추정 대신, 직전 1봉 수익률과 직전 2~24봉 평균 수익률로 축약한 3차원 가우시안 커널 밀도 추정(KDE) 조건부 전이 엔트로피 근사를 사용하며 후보 쌍 탐색 시 다중비교 보정은 적용하지 않는다(`src/ts/study2/causality.py`).
6. **연구 2 토론 주기·반사실 표본·일간 엣지 검증 상한 및 승인율 (제IV장 2.3~3.3절, 부록 A.3·B)**: 단일 로컬 GPU 환경에서 전 구간(`28,468`봉) 실행이 가능하도록 본 실행(`scripts/run_study2.py`)에서는 4봉마다 3라운드 토론을 수행하고 사이 봉은 직전 결정을 유지하며(`--debate-stride 4`, 시험 구간 토론 `4,931`봉), 근거 제거 반사실 토론은 시험 구간 토론 봉 2개마다 수행하고(`--counterfactual-every 2`, `2,466`쌍), 일간 LLM 인과 엣지 검증은 `BTC` 유입 후보를 우선하여 전이 엔트로피 상위 30개(`--edge-top-k 30`, 동일 ISO 주 내 검증 결과 재사용)로 상한을 두었다. 부록 B 원문 프롬프트를 그대로 적용한 결과 전체 일간 검증 결정 `35,610`건(`1,187`일) 중 `1,217`건(`3.42%`)만 승인되었으며, 당일 뉴스 헤드라인에 두 자산 간 직접적인 경제 경로가 나타나지 않아 승인된 엣지가 없는 날에는 하이퍼그래프 인용 없이 시장·뉴스 맥락만으로 토론이 진행된다.
7. **메타 강화학습 비교 모델 (제VI장 1.3절)**: `MAMLMetaPPOBaseline`(`src/ts/baselines/models.py`)은 2차 도함수나 메타 검증 손실 역전파를 계산하는 원형 MAML이 아니라, 태스크별 내부 적응 파라미터와 초기 파라미터의 차이를 선형 보간하는 Reptile 방식 1차 메타 갱신으로 구현되어 있다.
8. **모달리티 기여도 분석 (제VI장 4.2절, 표 6.9)**: 224차원 개별 특징에 대한 표본 추출식 KernelSHAP 대신, 시각(`128`)·기술(`64`)·감성(`32`) 3개 모달 블록의 `2^3 = 8`개 연합을 전수 평가하는 블록 단위 유한 Shapley 값을 각 봉의 활성 국면 전문가 풀과 선택 행동 로짓에 대해 계산한다(`src/ts/study1/attribution.py`).
9. **소거 실험 Variant A 정의 및 평가 단계 구분 (제III장 1.5절, 표 6.10, 표 6.12~6.13)**: `XGBoost` 국면 분류기와 3국면 동적 앙상블 구조는 Variant D와 동일하게 유지하고 상태 인코더 입력만 기술 지표 단일 모달(기본값 `--variant-a-modality tech_only`) 또는 뉴스 감성 제거(`visual_tech_no_senti`)로 제한한다. 변형마다 교사 LLM 토론을 다시 생성하는 비용을 분리하기 위해 표 6.10의 Variant A~C 및 표 6.12(`T`)·표 6.13(`K`) 민감도는 연구 1 앙상블 수준(`scripts/run_study1.py`)에서 평가하고, 전체 제안 모델(표 6.6~6.8)과 $\lambda_3$ 민감도(표 6.11)는 연구 3 증류 학생 수준(`scripts/run_study3.py`)에서 평가한다.
10. **연구 1 국면 라벨·마르코프 상태·횡보 벌점 `C`·전환 확인 `24`봉 보완 (제III장 2.1~3.3절, 표 A.1~A.2)**: 예행 실행에서 확인된 횡보 라벨 소멸과 과도한 회전율 문제를 완화하기 위해, 첫 학습 폴드(`2020-10`~`2021-09`) 내부 검증 분할만 사용하고 시험 구간(`2021-10`~`2023-12`)은 참조하지 않는 사전 등록 절차에 따라 다음 설정을 동결했다. (가) 단위근 로그 가격의 선형 추세 회귀에서 고정 `t = 2.0`을 쓰면 횡보 라벨이 소멸하므로 부록 표 A.2에 명시된 학습 폴드 내부 `60%` 분위수(`trend_quantile = 0.60`)로 임계값을 산출한다(`scripts/calibrate_study1_d15_fixes.py`). (나) 거래 비용이 직전 목표 비중 $w_{t-1}$에 의존하는 보상 구조와의 마르코프 정합성을 위해 정책·가치 신경망 입력에 $w_{t-1}$을 포함(`129`차원)하고 비연속 국면 구간 경계에서 회전율과 GAE 부트스트랩을 초기화한다. (다) 논문이 수치를 지정하지 않은 식 (III.19)의 횡보 국면 거래 벌점 계수 `C`는 첫 학습 폴드 내부 두 검증 구간(`2021-06~07`, `2021-08~09`) 검증(`scripts/calibrate_study1_sideways_penalty.py`, `runs/study1/fold1_val_sideways_penalty.json`)을 통해 `sideways_turnover_penalty = 0.05`로 동결했다. (라) `XGBoost` 국면 분류기는 역빈도 가중 없이 경험적 사전분포를 유지하고(`xgb_balance_classes = false`), 같은 두 내부 검증 구간 검증(`scripts/calibrate_study1_confirm_bars.py`, `runs/study1/fold1_val_regime_confirm.json`)을 통해 전환 임계(`0.60`) 연속 확인 봉 수를 `regime_confirm_bars = 24`로 동결했다(단, 시드 고정 상태에서도 GPU 연산 비결정성에 따른 실행 간 내부 검증 샤프 변동 폭이 약 `1.8`로 후보 간 격차와 유사하며 첫 검증 구간에서는 전 기간 횡보 판정이 유지되는 한계를 함께 기록한다). 또한 식 (III.22)의 PER 중요도 가중치는 영(0) TD 오차 이상치에 의한 기울기 소실을 막기 위해 상한 `10.0`과 미니배치 평균 자기정규화($\bar{\omega}=1$)를 적용하고 엔트로피 항에도 동일 가중치를 곱한다.
11. **연구 3 학생 온라인 실행 규칙·과제 손실 대안 검토·비용 민감도·기준 규칙 비교 및 단일 모델 잠재 공간 시각화 (제V장 3.2~4.2절, 제VI장 3~4절)**: (가) `Student1DCNN`은 직전 포지션을 입력받지 않고 봉마다 독립적으로 행동 확률 분포를 산출하므로, 단순 `argmax` 집행 시 현금(`0.0`)과 인접 매수(`+0.5`)의 확률이 비슷한 구간에서 잦은 포지션 전환이 발생한다. 이를 억제하기 위해 국면 분류기의 신뢰도 임계 규칙(식 III.16)과 동일하게 최대 행동 확률 $\max_a p_t(a) \ge \theta$일 때만 행동을 바꾸고 미달 시 직전 행동(초기값 현금 `0.0`)을 유지하는 `apply_action_confidence_gate`(`execute_student_weights`, `Study3Config.execution_mode = "gate"`, `action_confidence_threshold = 0.8`, `action_smoothing_span = 1`)를 온라인 집행에 적용하며, 학생 신경망 구조와 오프라인 증류 학습(손실 가중·`50` 에폭)은 변경하지 않는다. 이 실행 규칙의 도입 자체는 1회차 학생 본 실행(게이트 미적용 시 Phase 2 비용 전 샤프 `+0.59`, 비용 후 샤프 `-10.64`, 봉당 회전율 `0.123`)의 원인 분해 뒤에 결정되었고, 임계값 수치(`0.8`)만 첫 학습 폴드(`2020-10`~`2021-09`) 내부 두 검증 분할(`2021-06~07`, `2021-08~09`) 검증(`scripts/calibrate_study3_action_gate.py`, `runs/study3/fold1_val_action_gate.json`)으로 선택·동결했다(내부 검증에서 최대 행동 확률 평균이 약 `0.66`이어서 $\theta = 0.8$ 적용 시 검증 봉의 `96~99.7%`에서 현금을 유지하며, 본 실행 시험 구간에서도 `96.4%` 봉이 현금을 유지함). 이어서 행동 확률의 인과적 지수가중 평활(`smooth_action_probabilities`, 구간 $N \in \{1, 24, 72, 168, 336\}$)을 첫 학습 폴드 내부 세 분할로 추가 검증했으나(`scripts/calibrate_study3_action_gate.py --d19`, `runs/study3/fold1_val_action_smoothing.json`), 평활($N \ge 24$)과 임계값($\theta \ge 0.6$)을 결합한 12개 후보는 모두 세 검증 구간에서 거래가 한 번도 발생하지 않았고 실제 거래가 일어난 평활 후보는 모두 검증 샤프가 음수여서 평활 없는 `action_smoothing_span = 1`·`action_confidence_threshold = 0.8` 설정을 그대로 유지했다(무거래 동률 후보를 채택하지 않은 판단은 보고 수치를 유리하게 만들지 않으며 그 경위를 함께 기록한다). 아울러 식 (V.7)의 과제 손실을 연속 봉 기대 비중의 회전율 비용 차감 샤프 손실(`task_objective = "net_sharpe"`)로 교체하고 기대 비중 직접 집행(`execution_mode = "expected"`) 또는 보유 비중 기준 `0.25` 무거래 구간 집행(`"expected_band"`)과 결합하는 대안을 같은 첫 학습 폴드 내부 세 분할로 검증했으나(`scripts/calibrate_study3_task_objective.py`, `runs/study3/fold1_val_task_objective.json`), 기대 비중 직접 집행은 학습 구간 과적합으로 검증 평균 샤프가 `-17.62`(현행 `-0.58`)로 악화되었고 무거래 구간 집행은 세 검증 분할 모두에서 거래가 발생하지 않아 기본값(`task_objective = "ppo"`, `execution_mode = "gate"`)을 변경하지 않았다. 비교 모델(`Single PPO`, `정적 앙상블`, `Reptile식 메타 PPO`)에는 이 행동 전환 임계값을 적용하지 않으므로 표 6.6·6.7 비교 시 실행 규칙 차이가 존재한다. 선행 완료된 연구 2 산출물과의 설정 파일 해시(`config_sha256 = d16038cd1c2cbf63`) 정합성을 유지하기 위해 해당 값들은 `src/ts/config.py`의 `Study3Config` 기본값으로 둔다. (나) 학생 신호가 어느 거래 비용 수준까지 유지되는지 진단하기 위해 `scripts/run_study3.py`는 동일한 학생 의사결정(임계값 `0.8` 적용 `GATED` 및 단순 `argmax` `UNGATED`)을 Phase 2 수수료·변동성 슬리피지·시장 충격 계수에 `{0, 0.1, 0.25, 0.5, 1.0}` 배율을 곱한 비용 민감도(`CH6_COSTSENS_*`)로 함께 기록하며, `scripts/run_reference_rules.py`는 동일한 1봉 지연 회계와 Phase 2 비용 아래에서 학습 없는 단순 과거 가격 규칙(`24, 72, 168, 720`봉 이동평균 매수·현금 및 모멘텀 양방향, `CH6_REF_*`)을 평가하여 해당 비용 수준에서 생존 가능한 회전율 상한을 제시한다(이 기준 규칙들은 사후 진단 맥락용으로 보고된다). (다) 제VI장 잠재 공간 t-SNE 그림(`scripts/make_latent_figure.py`)은 9개 폴드별 학생 인코더의 은닉 벡터(`runs/study3/student_oos_hidden.npz`)를 함께 투영하면 시장 상태가 아니라 폴드 간 좌표 기저 차이가 군집으로 분리되므로, `2021`년 구간(`Fold 2` 학습 구간 `2021-01`~`2021-12`)만으로 학습된 단일 학생 모델(`LATENT_FIGURE_FOLD_ID = 2`)로 미관측 `2022`~`2023`년 전체 봉을 인코딩한 `runs/study3/student_single_model_hidden.npz`를 기본 입력으로 사용한다.
12. **부록 D 실시간 맥락 주입 설계**: `src/ts/future_work/film_double_buffer.py`는 향후 연구용 참조 구현으로 분리되어 있으며 제VI장 실험에는 사용되지 않는다.

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

# 2. 뉴스 헤드라인 DeepSeek-R1 3분류 로그확률 감성 채점 (로컬 Ollama 서버 필요, 재개 지원)
.\.venv\Scripts\python.exe tools/score_news_sentiment_llm.py

# 3. 연구 1: 9폴드 Walk-Forward, 표 6.9 모달 기여도, 표 6.10 소거(Variant A~D), 표 6.12 온도 T, 표 6.13 전문가 수 K
#    (실행 간 편차 확인용 반복 실행 시 --repeat-tag R2, --repeat-tag R3 지정 가능)
.\.venv\Scripts\python.exe scripts/run_study1.py

# 4. 연구 2: 동적 하이퍼그래프 및 근거 중심 LLM 토론 (로컬 Ollama 서버 필요, 체크포인트 재개 지원)
.\.venv\Scripts\python.exe scripts/run_study2.py --debate-stride 4 --counterfactual-every 2 --edge-top-k 30 --llm-backend ollama --llm-endpoint http://127.0.0.1:11434/api/chat --llm-model deepseek-r1:32b

# 5. 연구 3: 오프라인 삼중 손실 추론 증류, 성질 5.1 검증, 표 6.6~6.8 제안 모델, 표 6.11 lambda_3 민감도, 비용 민감도(CH6_COSTSENS_*), 표 6.15 일치율
.\.venv\Scripts\python.exe scripts/run_study3.py

# 6. 비교 모델(Single PPO, 정적 앙상블, Reptile식 메타 PPO) 및 Ledoit-Wolf 부트스트랩 유의성 검정
.\.venv\Scripts\python.exe scripts/run_baselines_and_eval.py

# 7. 동일 Phase 2 비용·1봉 지연 회계 기준 단순 과거 가격 기준 규칙 평가(CH6_REF_*)
.\.venv\Scripts\python.exe scripts/run_reference_rules.py

# 8. ledger/numbers.csv 기반 제VI장 마크다운 표 및 학생 잠재 공간 t-SNE 그림 생성
.\.venv\Scripts\python.exe scripts/make_tables.py
.\.venv\Scripts\python.exe scripts/make_latent_figure.py
```

전 단계를 순서대로 호출하는 오케스트레이터는 `scripts/run_full_pipeline.py`이며, 빠른 배선 점검 시 `--smoke-test`를 지정하면 자동으로 `TS_SMOKE_RUN=1`이 설정되어 최종 대장을 오염시키지 않는다.

---

## 5. 하드웨어와 소요 시간

로컬 측정 환경(Windows 11, NVIDIA GeForce RTX 5090 Laptop GPU 24GB VRAM, 시스템 RAM 31GB)에서 실측된 단계별 소요 시간과 자원 사용량은 다음과 같다.

- **WP0 기초 통계 및 학생 온라인 추론 지연 (`scripts/run_wp0_data_and_stats.py`, 커밋 `91c3a54`)**: `Student1DCNN.act_online` 단일 윈도우(`1 × 40 × 100`) 10,000회 반복 추론 지연 평균 `0.527 ms`, 중앙값 `0.462 ms`, 99백분위수 `1.462 ms`.
- **뉴스 헤드라인 DeepSeek-R1 감성 채점 (`tools/score_news_sentiment_llm.py`)**: 공개 뉴스 헤드라인 `50,008`건에 대한 로컬 Ollama `deepseek-r1:32b` 3분류 첫 토큰 로그확률 산출 약 `57분`.
- **연구 1 차트 렌더링 및 폴드 1 인코딩 메모리·시간 (`29,932`봉 전체 기준)**: `uint8` 캔들 차트 렌더링 `8.93초`(버퍼 `4.20 GiB`), 폴드 1(`8,741`봉) 사전학습 1에폭 + 학습·시험 인코딩 포함 총 `34.71초`, 프로세스 최대 작업 집합(Peak RSS) `6.05 GiB`(`6.49 GB`), GPU 최대 할당 메모리 `1.54 GiB`.
- **연구 1 폴드 1 내부 검증 (`scripts/calibrate_study1_rollouts.py`, `runs/study1/fold1_val_rollout_calibration.json`)**: 학습 `7,283`봉·검증 `1,458`봉 기준 `ResNet-18 + CBAM` 4에폭 사전학습 및 인코딩 `60.97초`, 기본 예산 `default_rollouts_per_fold = 64`에서 3국면 × 5전문가(15개 전문가) 온폴리시 `PPO + PER` 학습 `142.06초`(`2.37분`). 횡보 벌점·전환 확인 내부 두 분할 검증(`calibrate_study1_sideways_penalty.py`, `calibrate_study1_confirm_bars.py`)은 회당 약 `15분` 소요.
- **연구 1 9폴드 전체 본 실행 (`scripts/run_study1.py`, 커밋 `a3a45c0`·`36a8f27`, 로그 `runs/study1/metrics.json`)**: `BTCUSDT` `29,932`봉 차트 렌더링, 사전 채점된 `DeepSeek-R1` 감성 로짓 결합, 9개 Walk-Forward 폴드의 Variant D(`K_max = 7`)·Variant A(`K = 5`) 인코더 사전학습 및 `64`롤아웃 학습(`C = 0.05`, 전환 확인 `24`봉), 구간 II 3그룹 Shapley 기여도, 온도·전문가 수 민감도 평가를 모두 포함하여 회당 약 **`68분`** 소요(`--repeat-tag R2`·`R3` 반복 실행 동일, 직전 `FinBERT`·확인 `2`봉 설정 커밋 `f2953b7`에서는 `49분 11초`).
- **연구 2 후보 엣지 탐색·교사 지연 및 28,468봉 전체 본 실행 (`scripts/run_study2.py`, 커밋 `4d7fc79`, 로그 `runs/study2/metrics.json`)**: 60종 × 1,440봉(60일 창) 기준 일별 그레인저 + KDE 전이 엔트로피 탐색 약 `4.9초`. LLM 캐시를 끈 서로 다른 시험 구간 `50`개 봉의 교사 의사결정 총 지연(표 6.14) 평균 `16,835.8 ms`(`16.84초`), 중앙값 `11,027.4 ms`, 99백분위수 `36,758.9 ms`(구간별 평균: 뉴스 검색 `1.7 ms`, 전이 엔트로피 `3,101.4 ms`, 하이퍼그래프 갱신 `7.8 ms`, 3라운드 토론 `13,725.0 ms`). `49`종 자산·`28,468`봉 전체 순회(`--debate-stride 4 --counterfactual-every 2 --edge-top-k 30 --gpu-temp-limit 82`, 일간 그래프 갱신 `1,187`일·시험 구간 토론 `4,931`봉·반사실 `2,466`쌍)는 단일 노트북 GPU 온도 제한 조건에서 체크포인트 재개를 포함해 총 경과 약 `53시간`(`2026-11-13 00:29`~`2026-11-15 05:36`, 순 가동 약 `46시간`)에 완료되었다.
- **연구 3 9폴드 증류·$\lambda_3$ 민감도 및 비용 민감도 본 실행 (`scripts/run_study3.py`, 커밋 `891ef86`, 로그 `runs/study3/metrics.json`)**: 1분 봉 미시구조 윈도우 추출(`29,932`봉 × `10`윈도우), `FinBERT` 합의문 임베딩, Phase 2 9폴드 × $\lambda_3 \in \{0.0, \dots, 0.5\}$ 6개 설정(`50` 에폭) 학습, `Fold 2` 단일 학생 은닉 벡터 인코딩, Phase 2 비용 배율 민감도(`0`~`1`배) 및 Phase 1 5-폴드 교차검증을 모두 포함하여 약 **`2시간 8분`~`2시간 15분`** 소요(커밋 `891ef86` `134분 32초`, 커밋 `eee0a11` `128분 01초`, 1회차 커밋 `6930aa1` `129분 52초`). 폴드 1 내부 행동 게이트·평활 검증(`scripts/calibrate_study3_action_gate.py`)은 `3`~`5분`, 비용 차감 샤프 과제 손실 검증(`scripts/calibrate_study3_task_objective.py`)은 약 `17분` 소요.
- **비교 모델·유의성 검정, 기준 규칙 및 표·그림 생성 (`scripts/run_baselines_and_eval.py`, `scripts/run_reference_rules.py`, `scripts/make_tables.py`, `scripts/make_latent_figure.py`)**: Phase 1·Phase 2에 대한 Single PPO, 정적 앙상블(`1/N`), Reptile식 메타 PPO 학습·평가와 Ledoit-Wolf HAC 및 순환 블록 부트스트랩(`5,000`회) 유의성 검정에 약 **`24`~`29분`** 소요(커밋 `eee0a11` `24분 23초`, 커밋 `891ef86` `29분 04초`), 학습 없는 단순 가격 기준 규칙 평가(`419e919`) `2초` 미만, 제VI장 마크다운 표 생성 `1초` 미만, 단일 학생 잠재 공간(`17,519`봉) t-SNE 투영 및 `200`회 순열 검정에 **`47초`** 소요.
