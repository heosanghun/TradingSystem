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

## 상시
- `handoffs/`에 `검토 요청`이 올라오면 완료 조건을 직접 실행해 확인하고 `BOARD.md`를 갱신한다.
- 논문 수정 목록(실측값이 논문과 다른 곳, 구현이 서술과 다른 곳)을 `handoffs/CC_논문수정목록.md`에 누적한다.

## 요청 (다른 에이전트가 CC에게)

(없음)
