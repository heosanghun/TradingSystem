# 본 실행 절차 (대장에 최종 기록되는 실행)

세 에이전트가 같은 작업 폴더를 쓰기 때문에, 누군가 코드를 고치는 중이면 대장 게이트가 "미커밋 변경"으로 최종 기록을 막습니다. 그래서 본 실행은 **깨끗한 별도 체크아웃**에서 합니다. WP0는 이 방식으로 기록했습니다(커밋 `6d77ce8`).

## 준비 (이미 만들어져 있음)

- 체크아웃 위치: `C:\Project\TradingSystem_final` (git worktree)
- `data\`는 본 폴더의 `data\`로 연결된 정션이라 캐시를 함께 씁니다.
- 가상환경은 본 폴더의 `.venv`를 씁니다.

## 실행

```powershell
# 1. 실행할 커밋을 정한다(코드가 모두 커밋·검토된 상태).
cd C:\Project\TradingSystem_final
git fetch -q; git checkout -q --detach <커밋해시>
git status --short        # 아무것도 나오지 않아야 한다

# 2. 이 체크아웃의 코드가 쓰이도록 PYTHONPATH를 지정해 실행한다.
$env:PYTHONPATH = "C:\Project\TradingSystem_final\src"
C:\Project\TradingSystem\.venv\Scripts\python.exe scriptsun_study1.py <인자>

# 3. 결과를 본 폴더로 옮긴다.
#    ledger
umbers.csv 의 새 행, runs\<단계>\ 의 metrics.json과 로그
```

## 규칙

- 실행 전 `git status --short`가 비어 있어야 합니다. 대장에 `-dirty`나 `provisional` 사유가 남으면 그 실행은 최종이 아닙니다.
- 다음 단계의 입력이 되는 산출물(`runs/study1/study1_oos_signals.npz`, `runs/study2/teacher_signals.npz` 등)은 본 폴더의 `runs/`로 복사하고, 만든 커밋 해시를 인계 기록에 적습니다.
- 대장 병합은 CC가 합니다. 실행한 에이전트는 체크아웃의 `ledger/numbers.csv`를 인계 기록에 경로로 적어 둡니다.
- `.git`에 쓸 수 없는 에이전트(CX)는 1번의 `checkout`을 CC에게 요청합니다.
