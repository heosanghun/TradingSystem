# 본 실행 절차 (대장에 최종 기록되는 실행)

세 에이전트가 같은 작업 폴더를 쓰기 때문에, 누군가 코드를 고치는 중이면 대장 게이트가 "미커밋 변경"으로 최종 기록을 막습니다. 그래서 본 실행은 **깨끗한 별도 체크아웃**에서 합니다. WP0는 이 방식으로 기록했습니다(커밋 `6d77ce8`).

## 준비 (이미 만들어져 있음)

- 체크아웃 위치: `C:/Project/TradingSystem_final` (git worktree)
- 그 안의 `data/`는 본 폴더의 `data/`로 연결된 정션이라 캐시를 함께 씁니다.
- 가상환경은 본 폴더의 `.venv`를 씁니다.

## 실행

```powershell
# 1. 실행할 커밋을 정한다(코드가 모두 커밋·검토된 상태).
cd C:/Project/TradingSystem_final
git checkout -q --detach <커밋해시>
git status --short        # ledger/ 외에는 아무것도 나오지 않아야 한다

# 2. 이 체크아웃의 코드가 쓰이도록 PYTHONPATH를 지정해 실행한다.
$env:PYTHONPATH = "C:/Project/TradingSystem_final/src"
C:/Project/TradingSystem/.venv/Scripts/python.exe scripts/run_study1.py <인자>
```

3. 결과를 본 폴더로 옮깁니다: `ledger/numbers.csv`의 새 행, `runs/<단계>/`의 `metrics.json`과 로그.

## 규칙

- 실행 전 `git status --short`에 코드 변경이 없어야 합니다. 대장에 `-dirty`나 `provisional` 사유가 남으면 그 실행은 최종이 아닙니다.
- 다음 단계의 입력이 되는 산출물(`runs/study1/study1_oos_signals.npz`, `runs/study2/teacher_signals.npz` 등)은 본 폴더의 `runs/`로 복사하고, 만든 커밋 해시를 인계 기록에 적습니다.
- 대장 병합은 CC가 합니다. 실행한 에이전트는 체크아웃의 `ledger/numbers.csv` 경로를 인계 기록에 적어 둡니다.
- `.git`에 쓸 수 없는 에이전트(CX)는 1번의 `checkout`을 CC에게 요청합니다.
- 체크아웃 안의 파일은 고치지 않습니다. 수정은 본 폴더에서 하고 커밋한 뒤 다시 체크아웃합니다.

## R-2 (연구 2) 실행 명령

`C:/Project/TradingSystem_final`에서, `PYTHONPATH`를 그 체크아웃의 `src`로 두고 `TS_SMOKE_RUN`은 비운 채 실행합니다. Ollama 서버가 떠 있어야 합니다.

```
C:/Project/TradingSystem/.venv/Scripts/python.exe scripts/run_study2.py --debate-stride 4 --counterfactual-every 2 --edge-top-k 30 --llm-backend ollama --llm-endpoint http://localhost:11434/api/chat --llm-model deepseek-r1:32b
```

- 중단되면 같은 커밋에서 같은 명령을 다시 실행합니다. `runs/study2/checkpoint.sqlite`에서 이어집니다. 코드나 인자가 바뀌면 체크포인트 식별자가 달라져 처음부터 다시 돕니다(LLM 응답 캐시는 남아 있어 이미 받은 응답은 재사용).
- 진행 상황은 로그의 `[Study 2] bar ...` 줄(토론 50봉마다)에 남습니다.
