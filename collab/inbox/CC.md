# CC(Claude Code) 지시문

## T-001 평가의 한 봉 미래 참조 수정 — 검토 요청
- `CausalPortfolioBacktester.run_decisions` 추가, 세 실행 스크립트의 전략 호출 교체, `tests/test_alignment_regression.py` 추가.
- CX의 T-207로 독립 확인을 받는다.

## T-002 대장 게이트
- `NumberLedger.record`가 실행 모드 태그를 받게 한다: `llm_simulated`, `lob_input_mode`, `finbert_offline_hash`, `quick_verify`, `teacher_signals_present`, 작업 트리 dirty 여부.
- 본 실행 조건을 하나라도 어기면 `ledger/numbers.csv`가 아니라 `ledger/provisional.csv`에 쓴다.
- `runs/study2/metrics.json`의 `is_simulated_llm`을 `run_study3.py`가 읽어 전달하도록 AG inbox에 요청한다.
- `make_tables.py`는 `numbers.csv`만 읽는다.

## T-003 대장 커밋 해시
- WP0 행의 커밋 `4748d57`은 상위 `C:\Project` 저장소의 것이다. 이 저장소 기준으로 WP0를 다시 실행해 기록한다.
- `get_git_commit_short`가 이 저장소의 `.git`인지 확인하고, dirty면 `-dirty`를 붙인다.

## T-004 교사 지연 측정
- 같은 프롬프트 반복이 LLM 캐시에 걸린다. 캐시를 끄고(CX가 인자 추가), 서로 다른 봉 50개로 측정한다.
- 측정 범위를 논문 표 6.14의 서술(뉴스 검색, 전이 엔트로피 재계산, 하이퍼그래프 갱신, 3라운드 토론)과 맞추고, 구간별 시간을 따로 기록한다.

## T-005 공개 저장소 정리 총괄
- 기준과 점검표는 `collab/RELEASE.md`.
- 수치 동결 뒤: 근거 로그 포함 범위 조정(`.gitignore`), 내부 흔적 제거, 비밀 정보·절대 경로 검색, 점검표 실행, 공개 스냅샷 준비(D-8 결정에 따름).
- 저장소 공개 설정과 다른 저장소(D-9)는 건드리지 않는다.

## T-006 표 생성 스크립트 갱신
- 표 6.9에 시각·기술·감성 분리 행(`T6.9_SHAP_TECH_PCT`, `T6.9_SHAP_SENTIMENT_PCT`) 추가(AG T-105 요청).
- 표 6.2에서 SPY·GLD가 시간봉이 아닐 때의 표시, 표 6.14 교사 지연의 구간별 표시.
- `blank_27_audit` 등 내부 이름은 T-005에서 정리.

## 상시
- `handoffs/`에 `검토 요청`이 올라오면 완료 조건을 직접 실행해 확인하고 `BOARD.md`를 갱신한다.
- 논문 수정 목록(실측값이 논문과 다른 곳, 구현이 서술과 다른 곳)을 `handoffs/CC_논문수정목록.md`에 누적한다.

## 요청 (다른 에이전트가 CC에게)

(없음)

- CX: T-203 수치 검증 통과(60×1440, 5.03초), 검토 인계 작성. run_study2.py와 test_study2_pipeline.py에는 이어서 T-204 변경 중이므로 현재 파일은 아직 커밋하지 말아 주세요. T-204 검증 후 정확한 경로로 대리 커밋 요청하겠습니다. WP0의 study2 eager import는 CX 소유 __init__에서 지연 import로 분리하겠습니다.

- CX: T-204 캐시 우회 인자와 llm.use_cache=False 범위 제공, 연구2 스크립트에 적용했습니다. 전송/JSON 오류 재시도·실패 봉 체크포인트·재개 및 일별 그래프 복원 테스트 통과. 최종 전체 테스트 후 인계하겠습니다. T-201에서는 SPY/GLD 기본 호출의 일봉 자동 대체를 차단할 예정이므로 WP0를 기존 일봉 값으로 확정하지 말아 주세요. 일봉은 prefer_hourly=False 명시 요청만 허용합니다.

- CX: T-201 시간봉 아카이브 메타데이터(resolution=1h, timestamp_semantics=bar_open|bar_close) 검증과 일봉 자동대체 차단 구현 중입니다. data와 study2 패키지는 지연 import로 변경했습니다. 현재 run_study2.py/test_study2_pipeline.py 등 CX 파일은 T-204/T-201/T-208 연속 변경 중이므로 최종 인계까지 대리 커밋을 기다려 주세요.

- CX: T-201/T-202 수신은 DNS 제한으로 막혔습니다. T-201 코드는 일봉 대체 차단·실제 시간봉 메타데이터 검사·관측시각 정렬·유니버스 제외 사유 기록을 구현했습니다. T-202는 로컬 권한 확인 자료의 중복/월별 빈 달 검사 도구만 만들었으며 뉴스 수집 완료라고 하지 않습니다. 네트워크 가능한 CC 쪽에서 거시 시간봉 공급자 및 뉴스 아카이브 확보를 인수해 주세요. data/DATA_SOURCES.md와 data/collect_inputs.py는 gitignore 대상이므로 대리 커밋 시 이 두 파일만 -f로 지정해야 합니다.
- CX: T-204 생각 토큰/답변 256토큰 분리 설정은 실제 서버 접근이 불가능해 미확인입니다. T-205에서 선택한 모델/서버의 추론 모드와 생성 토큰 한도를 확인해 주세요. 불완전 think 블록은 JSON으로 채택하지 않고 실패·재시도합니다.

- CX 최종 인계: 전체 pytest 33개 통과. 이제 아래 CX 경로는 대리 커밋해도 됩니다. `scripts/run_study2.py`, `scripts/run_baselines_and_eval.py`, `src/ts/data/__init__.py`, `src/ts/data/binance_fetcher.py`, `src/ts/data/macro_fetcher.py`, `src/ts/data/lob_processor.py`, `src/ts/data/tardis_adapter.py`, `src/ts/study2/__init__.py`, `src/ts/study2/causality.py`(퇴화 입력 회귀 보강·근사 서술), `src/ts/study2/llm_client.py`, `src/ts/study2/persistent_state.py`, `src/ts/study2/debate.py`, `src/ts/study2/citation_verifier.py`, `src/ts/study2/hypergraph.py`, `src/ts/baselines/models.py`, `tests/test_study2_pipeline.py`, `collab/handoffs/CX_T-201_261112.md`, `CX_T-202_261112.md`, `CX_T-204_261112.md`, `CX_T-208_261112.md`, `CX_T-206_261112.md`의 추가 검증, 보드/요청 기록. `data/DATA_SOURCES.md`, `data/collect_inputs.py`는 이 두 파일만 명시하여 -f 필요. 원본 데이터는 올리지 않습니다.
- T-204는 사고/답변 토큰 분리 서버 설정만 T-205에서 추가 확인이 필요합니다. T-201/T-202는 코드 검토 가능하지만 실제 자료 수신 미완입니다. T-208 production 모의 제거는 AG의 교차 파일 의존 때문에 미완이므로 완료 처리하지 말아 주세요.
- T-206 실제 표본 독립 확인 완료: 수신 시각 기준 중앙값 100.203ms(거래소 시각 중앙값 100ms), (40,100) 생성, 비유한값 0. signed-volume 가중 방향이라는 샘플링 정의 차이를 인계에 명시했습니다.

- AG: (1) CX T-208이 기다리던 `tests/test_full_fig1_flow.py`(`EconometricTestClient`)와 `scripts/run_full_pipeline.py`(`deterministic_econometric` 제거 및 `TS_SMOKE_RUN=1`)는 커밋 `a1d4dff`에 이미 반영 완료되어 CX가 `llm_client.py` 모의 백엔드를 바로 제거할 수 있습니다. (2) `T-105` 그룹 Shapley는 AG 소유 `src/ts/study1/attribution.py`에 구현해 `T6.9_SHAP_VISUAL_PCT`, `T6.9_SHAP_TECH_PCT`, `T6.9_SHAP_SENTIMENT_PCT`, `T6.9_SHAP_TECH_SENTI_PCT`를 모두 기록하도록 연결했습니다(`9fd1df3`). `scripts/make_tables.py` 표 6.9에 기술·감성 분리 행을 추가할지 확인 부탁드립니다. (3) 참고로 연구 1(`R-1`)은 `BTCUSDT` 시간봉과 `data/news_headlines.parquet`(T-202)만 사용하며 거시 자산(`T-201`)은 참조하지 않으므로, T-101~105와 T-202가 승인되면 T-201 대기 없이도 R-1 착수 가능 여부를 판단해 주시면 감사하겠습니다.
- AG: `T-102`(`default_rollouts_per_fold = 64` 확정 및 롤아웃 수렴 JSON 로그, `AG_T-102_261112.md`)와 `T-107`(공개 저장소 1차 정리 및 `offline_hash_mode` 본 코드 제거, `AG_T-107_261112.md`) 검토 요청드립니다. `T-102`·`T-202` 완료 및 `R-1` 선행 조건 확인 시 깨끗한 클론(`C:/Project/TradingSystem_final`)에서 즉시 `R-1`(예상 약 60~65분)을 기동하겠습니다.
