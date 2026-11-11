# 에이전트 공통 안내

이 저장소는 세 에이전트(CC = Claude Code, AG = Antigravity, CX = Codex)가 함께 작업합니다. 작업을 시작하기 전에 반드시 다음 순서로 읽으십시오.

1. `collab/README.md` — 공동 작업 규약
2. `collab/ROLES.md` — 역할과 파일 소유권
3. `collab/BOARD.md` — 작업 보드
4. `collab/inbox/<자기 이름>.md` — 자기 지시문 (`AG.md`, `CX.md`, `CC.md`)
5. `collab/DECISIONS.md` — 사용자 결정 목록

핵심 규칙
- 자기 소유 경로만 수정하고, 자기 경로만 `git add <경로>`로 커밋합니다.
- 스모크·모의 실행의 수치를 `ledger/numbers.csv`에 쓰지 않습니다.
- 사후 보정 코드를 넣지 않고, 결과는 나온 대로 보고합니다.
- 모델이 봉 t 종가에서 정한 비중의 백테스트는 `CausalPortfolioBacktester.run_decisions`로만 합니다.
- 실행 환경: `.\.venv\Scripts\python.exe`, 테스트는 `.\.venv\Scripts\python.exe -m pytest -q`
