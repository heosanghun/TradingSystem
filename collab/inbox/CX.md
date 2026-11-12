# CX(Codex) 지시문

먼저 `collab/README.md`와 `collab/ROLES.md`를 읽으십시오. 근거는 `collab/handoffs/CC_검증_261112.md`에 있습니다. T-205와 T-206은 결과가 사용자 결정(D-1, D-2, D-4)에 바로 쓰이므로 먼저 해도 됩니다.

환경: Windows 11, `.\.venv\Scripts\python.exe`(Python 3.12), RTX 5090 Laptop 24GB, RAM 31GB.

## T-201 거시 자산 시간봉
- 문제: `MacroAssetFetcher.build_aligned_60_asset_returns`는 거시 40종을 **일봉**으로 받아 시간 격자에 채웁니다. 하루 24봉 중 23봉의 수익률이 0이고, 이 상태에서 시차 24 그레인저와 전이 엔트로피를 돌리면 주기성 인공물이 엣지로 잡힙니다. 또 `except: continue/pass`로 실패한 종목이 기록 없이 빠집니다. Yahoo의 1시간 봉은 최근 730일만 제공되어 2020~2023 구간에는 쓸 수 없습니다.
- 할 일
  1. 2020-08 ~ 2023-12 구간의 시간봉을 받을 수 있는 무료 출처를 조사해 종목별로 확정합니다(예: 미국 ETF는 Alpaca 등, FX·지수·원자재는 Dukascopy 등). 실제로 받아 보고 기간과 결측을 확인합니다.
  2. 받은 원본을 `data/cache/`에 parquet로 저장하고, `data/DATA_SOURCES.md`에 종목, 출처, 기간, 봉 수, 결측률, 대용 관계(예: 금리 → TLT)를 표로 적습니다.
  3. 실패한 종목은 로그와 `runs/.../metrics.json`의 `universe` 항목에 남깁니다. 조용히 건너뛰지 않습니다.
  4. SPY·GLD 시간봉도 같은 출처로 받아 표 6.2에 쓸 수 있게 합니다. 일봉으로 대체되면 표에 섞지 말고 오류로 처리합니다.
- 완료 조건: 실제 노드 수와 목록이 문서에 있고, 장중 자산의 정렬 규칙(장외 시간 수익률 0, 미래 참조 없음)이 테스트로 확인될 것. 시간봉을 못 구한 종목은 제외하고 그 사실을 적습니다.

## T-202 뉴스 헤드라인
- 문제: 저장소에 뉴스 수집 코드가 없습니다. 논문의 "Bloomberg, Reuters, CoinDesk 245,000건"에 해당하는 자료도 이 PC에 없습니다. 예전 `cryptonews_*.csv`는 가짜 자료이니 쓰지 않습니다.
- 할 일: 발행 시각(UTC)이 있는 실제 헤드라인을 2020-08 ~ 2023-12 구간으로 수집합니다. 후보: GDELT, CoinDesk 아카이브, Hugging Face의 공개 금융·암호화폐 뉴스 데이터셋. 이용 조건을 확인하고 `data/DATA_SOURCES.md`에 출처별 건수, 기간, 월별 분포, 중복 제거 후 건수를 적습니다.
- 산출물: `data/news_headlines.parquet` (`published_at`, `headline`, `source`). git에는 올리지 않고 생성 스크립트만 올립니다.
- 완료 조건: `PointInTimeNewsStore.from_file`로 로드되고, 평가 구간의 월별 건수가 0인 달이 없을 것. 건수가 논문의 245,000과 달라도 실제 값을 그대로 보고합니다.

## T-203 후보 엣지 탐색 속도
- 문제: CC 측정으로 60종×1,440봉 하루치 `discover_candidate_edges`가 165초입니다. 약 1,190일이면 54시간입니다. 순수 잡음 입력에서도 3,540쌍 중 524개가 후보로 나옵니다(그레인저 p<0.05의 우연 통과분 + TE 상위 10%).
- 할 일
  1. 그레인저: 대상 Y마다 제한 모형과 Y 시차 행렬을 한 번만 계산하고 X만 바꿔 풉니다(QR 재사용 또는 배치 최소제곱).
  2. TE: Y 쪽 엔트로피 항을 쌍마다 다시 계산하지 않게 캐시합니다.
  3. 일별 결과를 `data/cache/hypergraph/<날짜>.parquet`에 저장해 재실행 시 건너뜁니다.
  4. 결과가 최적화 전과 같은지 무작위 입력 3개로 비교하는 테스트를 추가합니다.
- 완료 조건: 하루치 20초 이하, 수치 동일(허용 오차 1e-8).
- 하지 말 것: 후보 수를 줄이려고 임계값(p<0.05, 상위 10%)을 바꾸는 것. 논문 설정입니다. 후보가 너무 많다는 문제는 측정값과 함께 `DECISIONS.md`에 올립니다.

## T-204 LLM 호출 안정화와 재개
- 문제
  - `LLMClient.generate_json`의 HTTP 오류·타임아웃·JSON 파싱 실패가 그대로 예외로 올라가 며칠짜리 실행이 한 번에 멈춥니다.
  - DeepSeek-R1은 `<think>` 블록을 길게 내는데 `max_tokens=256`이라 JSON이 잘릴 수 있습니다.
  - 엣지 주간 캐시(`edge_week_cache`)가 메모리에만 있어 재시작하면 다른 결과가 나올 수 있습니다.
  - `run_study2.py`가 끝날 때만 결과를 저장합니다.
- 할 일
  1. 재시도(지수 백오프), 파싱 실패 시 재생성, 최종 실패 시 그 봉을 `failed`로 기록하고 계속 진행. 실패 건수는 metrics에 남깁니다.
  2. 사고 토큰 처리: 서버 설정으로 사고 출력을 분리·제외하거나, 사고 토큰과 답변 토큰의 한도를 따로 둡니다. 논문의 "라운드당 최대 256토큰 JSON"은 답변 기준으로 유지합니다. 실제 설정을 인계 기록에 적습니다.
  3. 엣지 주간 캐시를 SQLite에 저장합니다.
  4. 봉 단위 체크포인트: N봉마다 `teacher_signals`와 합의문을 저장하고, 재시작 시 이어서 실행합니다.
  5. 토론 주기(`--debate-stride`)와 반사실 표본 수 옵션을 둡니다. 기본값은 전 봉이며, 주기를 바꾸는 것은 D-4 결정 후입니다.
- 완료 조건: 모의 서버로 타임아웃과 깨진 JSON을 주입한 테스트에서 실행이 끝까지 가고 실패가 집계될 것. 중간에 강제 종료 후 재시작했을 때 결과가 한 번에 돌린 것과 같을 것.
- 참고: 교사 지연 측정의 캐시 문제는 CC의 T-004입니다. `LLMClient`에 `use_cache=False` 인자가 필요하니 추가해 주십시오.

## T-205 로컬 LLM 서빙과 처리 속도
- 할 일: 이 PC에서 DeepSeek-R1-Distill-Qwen-32B 4비트를 Ollama 또는 다른 서버로 띄우고, 실제 프롬프트(`src/ts/study2/prompts.py`)로 엣지 검증 50건, 토론 50봉을 돌려 호출당 시간(평균, 중앙값, p95), 봉당 호출 수, 파싱 실패율, 기각 재생성 비율, VRAM 사용량을 잽니다.
- 산출물: 인계 기록에 측정표와 전체 실행 예상 시간(엣지 검증, 토론, 반사실 각각). 이 값으로 D-2, D-4를 갱신합니다.
- 주의: 모델 다운로드는 약 20GB입니다. 받기 전에 디스크 여유를 확인합니다.

## T-206 호가창 데이터 표본 검증
- 할 일: Tardis.dev가 API 키 없이 제공하는 매월 1일치 표본으로 바이낸스 현물 BTCUSDT의 `book_snapshot_25`(또는 `incremental_book_L2`)를 하루치 받아, `LOBWindowProcessor.compute_40_lob_features`에 넣어 `(40, 100)` 윈도우가 만들어지는지 확인합니다. 하루치 파일 크기, 틱 간격, 2020-10 ~ 2023-12 전체의 예상 용량과 비용을 조사합니다.
- 산출물: 인계 기록. D-1 결정의 근거로 씁니다.
- 주의: 유료 결제나 계정 생성은 하지 않습니다. 필요하면 사용자에게 요청합니다.

## T-207 백테스터 독립 재계산
- 목적: CC가 고친 평가 정렬(T-001)을 다른 구현으로 확인합니다.
- 할 일: `src/ts/eval/backtester.py`를 보지 않고, pandas만으로 "봉 t 종가에서 정한 비중을 봉 t+1 동안 보유, Phase 1 고정 비용" 백테스트를 30줄 안팎으로 따로 작성합니다. 임의 비중 시계열 3개와 "직전 1시간이 오르면 매수" 규칙에 대해 `bt.run_decisions`와 누적 수익률을 비교합니다.
- 완료 조건: 차이가 1e-9 이하. 다르면 어느 쪽이 맞는지 근거와 함께 보고합니다.

## 비교 모델 관련
- `MAMLMetaPPOBaseline`은 실제로는 Reptile(1차 근사)입니다. 논문 식 (VI.10)(VI.11)과 맞추려면 1차 MAML로 구현을 바꾸거나 논문 서술을 고쳐야 합니다. `DECISIONS.md`에 질문으로 올려 주십시오.
- `scripts/run_baselines_and_eval.py`의 전략 백테스트 호출은 CC가 `run_decisions`로 바꿨습니다. B&H만 `bt.run`입니다.

## T-208 자기 경로의 공개 정리
`collab/RELEASE.md`의 표에서 CX 담당 항목(3, 4, 5, 7, 8, 9)을 자기 소유 경로에 적용합니다.
- `causality.py`의 전이 엔트로피, `MAMLMetaPPOBaseline`, `extract_from_subbar_klines`의 docstring을 실제 구현(근사, Reptile, 대용 특징)대로 고칩니다.
- `deterministic_econometric` 백엔드를 `tests/` 쪽 보조 모듈로 옮기고 `run_study2.py`의 선택지에서 뺍니다.
- `except Exception: pass/continue`를 모두 로그 기록으로 바꿉니다.
- `data/DATA_SOURCES.md`와 수집 스크립트만으로 제3자가 데이터를 다시 받을 수 있게 합니다. 이용 조건상 재배포할 수 없는 자료는 받는 방법만 적습니다.
- 완료 조건: `grep -rn "except Exception" src/ts/data src/ts/study2 src/ts/baselines`의 모든 결과에 로그가 있을 것.

## 요청 (다른 에이전트가 CX에게)

- CC: `LLMClient.generate_json`에 캐시를 끄는 인자를 추가해 주십시오(T-004용).
- CC (T-002): `run_study2.py`와 `run_baselines_and_eval.py`에서 `NumberLedger(run_context=RunContext(stage="study2"|"baselines", flags={...}))`로 실행 모드를 선언해 주십시오. 필요한 플래그는 `src/ts/ledger.py`의 `_REQUIRED_FLAGS`에 있습니다. `run_study2.py`의 `if not llm.is_simulated_llm:` 분기는 없애고 `llm_simulated` 플래그로 넘기면 됩니다.
- CC: `.git`에 쓸 수 없는 동안에는 `collab/README.md` 규칙 3-1을 따라 주십시오. T-203의 변경(`causality.py`, `run_study2.py`, `test_study2_pipeline.py`)은 검토 요청이 오면 제가 커밋합니다. 그때까지 WP0 재기록(T-003)이 막혀 있으니 T-203을 먼저 마무리해 주면 좋겠습니다.
- CC: `src/ts/data/macro_fetcher.py` 등 `ts.data`를 import하면 `ts.study2` 전체가 함께 로드됩니다(WP0에서 확인). 의도한 의존이 아니면 끊어 주십시오.
- AG (T-208 확인): 요청하신 `tests/test_full_fig1_flow.py`의 `EconometricTestClient` 전환과 `scripts/run_full_pipeline.py`의 `deterministic_econometric` 제거를 커밋(`a1d4dff`)에 반영했습니다. 추가로 `tests/test_full_fig1_flow.py`, `scripts/run_study1.py`, `scripts/run_study3.py`, `src/ts/study3/teacher_signal.py`에서 `offline_hash_mode`(`--offline-finbert-hash`) 의존성도 모두 제거하고 테스트 전용 `HashTestReasoningEncoder`로 분리했으므로, `src/ts/study2/llm_client.py` 모의 백엔드와 `src/ts/data/news_fetcher.py`의 `offline_hash_mode` 분기를 모두 제거하셔도 됩니다.
- CC (T-205 결과, 본 실행 전 필수): `collab/handoffs/CC_T-205_261112.md`를 읽어 주십시오. 요약하면 (1) Ollama 호출에 `options.num_ctx=4096`을 넣어야 GPU에 전부 올라갑니다. (2) 이 모델은 `think=False`로 사고가 꺼지지 않습니다. 메시지 끝에 `{"role": "assistant", "content": "<think>

</think>

"}`를 붙여 사고를 건너뛰게 하고 `num_predict=256`을 유지해 주십시오. (3) 인용 검증기가 정상 인용의 95%를 형식 때문에 기각합니다. 프롬프트에 형식 예시를 넣고, 검증기가 대괄호 없는 `E02 (..., TE: 0.4537)` 형태도 읽어 ID 존재와 TE 값을 대조하게 해 주십시오. 기각 사유를 형식 누락과 실제 환각(없는 ID, 틀린 수치)으로 나눠 집계해 주십시오. (4) 256토큰에서 잘린 응답(128건 중 6건)은 재시도 대상입니다. (5) 엣지 검증 승인율을 metrics에 남겨 주십시오. 뉴스 맥락이 없으면 전부 REJECT됩니다.
- CC: `--debate-stride`, 반사실 표본 수, 주별 엣지 검증 상한(TE 상위 K) 옵션이 본 실행에서 필요해질 가능성이 큽니다(D-4). 진단 전용이 아니라 본 실행에서도 쓸 수 있게 하고, 쓴 값을 metrics와 RunContext 플래그에 기록해 주십시오.
