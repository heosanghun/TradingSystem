# 역할과 파일 소유권

역할은 "누가 그 코드를 가장 잘 아는가"와 "서로의 결과를 독립적으로 검증할 수 있는가"로 나눴습니다. 구현한 쪽과 검증하는 쪽을 분리합니다.

## CC (Claude Code) — 총괄, 평가 정합성, 검증

- 작업 보드와 지시문 관리, 인계 검토와 승인
- 평가 코드의 정확성: 백테스터 정렬, 비용 모형, 지표, 유의성 검정
- 출처 대장과 표 생성, 논문 수치 대조
- 시점 정합(미래 참조) 테스트

소유 경로
- `collab/` (각자 쓰는 칸 제외)
- `src/ts/eval/`, `src/ts/ledger.py`, `src/ts/data/temporal_guard.py`
- `scripts/make_tables.py`, `scripts/run_wp0_data_and_stats.py`
- `tests/test_eval_and_tables.py`, `tests/test_temporal_alignment.py`, `tests/test_alignment_regression.py`
- `ledger/`, `구현계획_261111.md`, `AGENTS.md`, `CLAUDE.md`

## AG (Antigravity) — 연구 1·3 구현과 본 실행

- 연구 1(인코더, 국면 분류, PPO 앙상블)과 연구 3(증류, 학생)의 구현 보완
- 연구 1 → 3 본 실행과 학습 로그 관리
- 통합 실행기

소유 경로
- `src/ts/study1/`, `src/ts/study3/`, `src/ts/future_work/`
- `scripts/run_study1.py`, `scripts/run_study3.py`, `scripts/run_full_pipeline.py`
- `tests/test_study1_pipeline.py`, `tests/test_study3_distillation.py`, `tests/test_full_fig1_flow.py`
- `README.md`, `configs/`

## CX (Codex) — 데이터, 연구 2, 비교 모델

> 2026-11-12 사용자 지시로 연구 2의 코드 수정과 본 실행(R-2)은 CC가 인수했다. CX 샌드박스에서는 로컬 LLM 서버에 접속할 수 없기 때문이다. `src/ts/study2/`, `scripts/run_study2.py`, `tests/test_study2_pipeline.py`는 R-2가 끝날 때까지 CC가 고친다. CX는 이 파일들을 고치기 전에 CC inbox에 먼저 적는다. 비교 모델(R-4)과 T-208의 나머지는 그대로 CX 담당이다.

- 실데이터 확보: 거시 자산 시간봉, 뉴스, 호가창 가능성 조사
- 연구 2(인과 탐색, 하이퍼그래프, LLM 토론)의 성능과 안정성
- 로컬 LLM 서빙과 처리 속도 측정
- 비교 모델

소유 경로
- `src/ts/data/` (`temporal_guard.py` 제외)
- `src/ts/study2/`, `src/ts/baselines/`
- `scripts/run_study2.py`, `scripts/run_baselines_and_eval.py`
- `tests/test_study2_pipeline.py`
- `data/` 아래 수집 스크립트와 데이터 설명 문서

## 공유 파일

- `src/ts/config.py`, `pyproject.toml`: 수정 전에 `BOARD.md`에 한 줄 적고, 한 번에 한 명만 고친다.
- `runs/`: 각자 자기 스크립트의 출력 폴더만 쓴다. git에는 `metrics.json`과 로그 요약만 올린다.

## 교차 검증

| 만든 쪽 | 검증하는 쪽 |
|---|---|
| AG의 연구 1·3 | CC |
| CX의 데이터·연구 2 | CC, 실행 가능성은 AG가 통합 실행으로 확인 |
| CC의 평가 코드 | CX가 독립 재계산으로 한 번 확인(T-207) |
