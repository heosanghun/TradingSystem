# 지능형 트레이딩 시스템 실측 통합 코드베이스 (`TradingSystem`)

박사학위논문 개정본 **`[개정본] 박사논문_개정본_본보기_261110.pdf`** (제I~VII장, 부록 A~D) 및 전체 시스템 구조도(**`doc/논문수정 가이드/4_fig1/fig1_3_system_overview_academic.png`**)와 1:1 대응하도록 처음부터 끝까지 클린 재작성된 **사후 보정 무결성(Zero Post-Hoc Calibration)** 실측 코드베이스입니다.

---

## 1. 핵심 설계 원칙
1. **사후 보정 코드 완전 배제**: 기존 레포지토리에 존재하던 `paper_alignment`, `calibrate_to_thesis`, 미래 수익률 참조, `.item()` 그래디언트 단절 버그를 100% 제거하였습니다.
2. **시점 인과성 강제 (`src/ts/data/temporal_guard.py`)**: 모든 Walk-Forward 폴드(`12개월 학습 -> 3개월 시험`, 총 9폴드), Trend Scanning 전방 창 퍼징(`t + L < train_end`), 뉴스 발행 시각 필터링(`published_at < bar_close_utc`)을 런타임 어설션으로 검증합니다.
3. **27개 `"측정 후 기입"` 빈칸 전수 측정**: 표 6.8(3칸), 표 6.11(6칸), 표 6.12(5칸), 표 6.13(9칸), 표 6.15(4칸) 및 표 6.2~6.14 전체 지표를 `ledger/numbers.csv` 출처 대장과 함께 자동 산출합니다.

---

## 2. 디렉토리 및 아키텍처 매핑

```text
c:\Project\TradingSystem\
├── configs/
│   └── appendix_a_config.json       # 논문 부록 A (표 A.1~A.4) 및 제VI장 설정값 (JSON)
├── src/ts/
│   ├── config.py                    # 불변 Dataclass 설정 로더 + SHA-256 해시 검증
│   ├── ledger.py                    # 숫자 출처 대장 (ledger/numbers.csv) 기록기
│   ├── data/                        # 입력 ①~④ 데이터 수집 및 시점 인과성 검증기
│   │   ├── temporal_guard.py        # 미래 정보 누수 런타임 차단기
│   │   ├── binance_fetcher.py       # 입력 ① BTC/USDT 1h·1m 봉 수집 및 결측 봉 집계
│   │   ├── news_fetcher.py          # 입력 ② 금융 뉴스 중복 제거 및 시점 필터 + EWMA(α=0.08)
│   │   ├── macro_fetcher.py         # 입력 ③ ~60개 암호화폐·거시 자산 수익률 및 SPY/GLD 대조군
│   │   └── lob_processor.py         # 입력 ④ 100틱 × 40특징 호가창/체결 미시구조 전처리기
│   ├── study1/                      # 연구 1 (제III장): 다중 모달 국면 적응형 앙상블 강화학습
│   │   ├── chart_renderer.py        # 60봉 캔들+거래량 224×224×3 차트 렌더러 (양봉 적색/음봉 청색)
│   │   ├── encoders.py              # ResNet-18+CBAM(128) + 기술지표(64) + 감성(32) + 2토큰 크로스 어텐션
│   │   ├── regime_labeling.py       # SMA-50(Phase 1) & 퍼징된 Trend Scanning(Phase 2) 라벨러
│   │   ├── regime_classifier.py     # XGBoost (300 trees, depth 6) + θ=0.60 전환 임계 게이트
│   │   ├── per_buffer.py            # SumTree 우선순위 경험 재생 (α=0.6, β: 0.4→1.0)
│   │   ├── ppo_expert.py            # 국면별 보상 함수(상승/하락 Sortino/횡보) PPO 전문가
│   │   └── ensemble.py              # 30일(720봉) 롤링 샤프 볼츠만 동적 가중 앙상블 (T=10, K∈{3,5,7})
│   ├── study2/                      # 연구 2 (제IV장): 동적 하이퍼그래프 & 근거 중심 LLM 토론
│   │   ├── causality.py             # 그레인저 F-검정(lag=24) + 가우시안 KDE 정규화 전이 엔트로피
│   │   ├── prompts.py               # 부록 B 영문 JSON 시스템/유저 프롬프트 (4개 역할)
│   │   ├── llm_client.py            # vLLM/Ollama DeepSeek-R1-Distill-Qwen-32B 클라이언트 + SQLite 캐시
│   │   ├── hypergraph.py            # 스펙트럴 군집화 하이퍼그래프 H_t 및 랜덤 워크 전이 행렬 P
│   │   ├── citation_verifier.py     # [Edge ID / TE] 인용 정합성 검증 및 환각 기각기
│   │   └── debate.py                # 알파 분석가 ⇄ 리스크 관리자 3라운드 토론 + 메타 PM(β=8, γ=6)
│   ├── study3/                      # 연구 3 (제V장): 이중 프로세스 추론 증류 및 온라인 학생 신경망
│   │   ├── teacher_signal.py        # 소프트 타깃 P_teacher(σ_w=0.25, τ=2.0) + FinBERT [CLS](768) + 10창 정렬
│   │   ├── student_model.py         # 부록 C 27줄 정확 일치 Student1DCNN + 온라인 단독 실행 경로(act_online)
│   │   └── distillation.py          # 3중 손실(L_task + L_policy + L_reasoning) + 성질 5.1 직교 그래디언트 검증
│   ├── baselines/
│   │   └── models.py                # Buy & Hold, Single PPO, Static Ensemble(1/N), MAML Meta-PPO
│   ├── eval/                        # 제VI장 평가 프로토콜 및 통계 검정
│   │   ├── cost_model.py            # Phase 1 고정(0.05%) & Phase 2 ATR(14)+Almgren-Chriss 동적 비용(식 VI.8)
│   │   ├── metrics.py               # 표 6.2 통계/ADF/Ljung-Box, 식 VI.3~VI.5 성과, 표 6.8 위기 구간 손실
│   │   ├── walk_forward.py          # 9폴드 롤링 Walk-Forward (12m 학습 → 3m 시험) 분할기
│   │   ├── backtester.py            # 인과적 1시간 리밸런싱 포트폴리오 시뮬레이터
│   │   ├── significance.py          # Ledoit & Wolf (2008) HAC + 순환 블록 부트스트랩(24봉, 5,000회)
│   │   ├── shap_analysis.py         # KernelSHAP 모달리티 기여도 분석 (식 VI.12, 표 6.9)
│   │   ├── latency_benchmark.py     # CUDA 동기화 10,000회 추론 지연 측정기 (표 6.14)
│   │   └── xai_metrics.py           # 표 6.15 설명 가능성 5대 정량 지표 산출기
│   └── future_work/
│       └── film_double_buffer.py    # 부록 D 향후 연구용 FiLM + 이중 버퍼 공유 메모리 참조 설계 (제VI장 미사용)
├── scripts/                         # 단계별 실측 실행 및 표 자동 생성 스크립트
└── tests/                           # 전체 모듈 단위/통합 테스트 스위트 (pytest)
```

---

## 3. 샌드박스 실행 방법 (`.venv`)

모든 명령어는 로컬 가상환경 `c:\Project\TradingSystem\.venv\Scripts\python.exe` 내부에서 실행합니다.

```powershell
# 1. 단위·통합 테스트 전체 검증
.\.venv\Scripts\pytest.exe -q

# 2. WP0: 바이낸스 BTC/USDT + SPY/GLD 실측 통계(표 6.2), ADF/Ljung-Box, B&H, 학생 10,000회 지연(표 6.14)
.\.venv\Scripts\python.exe scripts/run_wp0_data_and_stats.py

# 3. WP2 (연구 1): 9폴드 Walk-Forward, 표 6.9 SHAP, 표 6.10 소거(A~D), 표 6.12 온도 T, 표 6.13 전문가 수 K
.\.venv\Scripts\python.exe scripts/run_study1.py

# 4. WP3 (연구 2): 60자산 하이퍼그래프 + DeepSeek-R1-Distill-Qwen-32B 토론 + 표 6.15 인용/환각/근거제거 실험
#    (사전에 로컬 vLLM 또는 Ollama 서버 기동 필요; 오프라인 파이프라인 점검 시 --llm-backend deterministic_econometric)
.\.venv\Scripts\python.exe scripts/run_study2.py --llm-backend openai_compat --llm-endpoint http://localhost:8000/v1/chat/completions

# 5. WP4 (연구 3): FinBERT + Student1DCNN 오프라인 추론 증류 + 표 6.11 λ_3 민감도(0.0~0.5) + 표 6.8/6.15 학생 실측
.\.venv\Scripts\python.exe scripts/run_study3.py

# 6. WP1 & WP5: 비교 모델(Single PPO, Static Ensemble, MAML) + 표 6.8 위기 손실 + Ledoit-Wolf 부트스트랩 검정
.\.venv\Scripts\python.exe scripts/run_baselines_and_eval.py

# 7. 제VI장 전체 표(표 6.2~6.15) 마크다운 및 27개 빈칸 점검 리포트 자동 생성
.\.venv\Scripts\python.exe scripts/make_tables.py
```
