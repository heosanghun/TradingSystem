# 학위논문 성과 검증 자료

BTC/USDT 원 시스템과 추세·회복 현물 전략을 같은 27개월 비용 조건에서 비교한 자료다. 숫자의 출처와 재현 범위를 아래에 정리했다.

| 먼저 볼 자료 | 확인하는 내용 |
|---|---|
| [성과 비교와 해석](FINDINGS.md) | 원고 표 6.7 인쇄값, 원 시스템 실측, 탐색 전략, 반례 |
| [시간별 검산](verify_hourly.py) · [관측 파일](hourly/) | 19,727봉 및 24,840봉의 결정 비중과 순수익률로 다섯 성과지표 재계산 |
| [재학습·민감도 로그](logs/verify_p61_examiner_output.txt) · [회귀 테스트 로그](logs/pytest_output.txt) | 별도 경로에서 재학습한 시드 42 텐서 18개와 시간별 테이블 4개 일치, 테스트 62개 통과 |
| [증류 소거 결과](results/ablation.csv) · [실험 코드](methods/distillation_ablation.py) | 같은 학습 라벨과 진입일에서 시드 7·42·97의 교사 항 포함·제거·교란 비교 |
| [폴드·기본 결과](results/metrics.json) · [강건성](results/audit.json) · [후보 선택 보정](results/selection_audit.json) · [별도 시장](results/bitstamp_rule.json) | 9개 워크포워드 폴드와 후속 고정 모델, 비용·지연·표본 집중, 선택 보정 구간, Bitstamp 규칙 대조 |
| [진입 조건](../src/ts/baselines/trend_entry_student.py) · [학습·평가](../src/ts/baselines/momentum_condition_student.py) · [체결 엔진](../src/ts/eval/backtester.py) | 일봉에서 조건 생성, 허용일에 연속 비중 결정, 다음 시간봉부터 적용 |

## 내려받은 자료로 직접 확인하기

저장소 루트에서 Python 3.10 이상으로 다음 명령을 실행한다. 표준 라이브러리만 사용하며 네트워크와 GPU가 필요 없다.

```shell
python review/verify_hourly.py
```

검산기는 파일별 SHA-256을 점검하고 각 기간의 전체 시각과 현물 비중 범위, 일중 비중 변경 여부를 확인한다. 이어 시간별 순수익률에서 소르티노·샤프·누적수익률·연복리·최대낙폭을 계산해 보관값과 절대오차 1e-8 이내인지 판정한다. 두 기간 × 세 대조 조건 × 다섯 지표, 총 30개를 검사한다. [기대 출력](logs/hourly_recalculation.txt)

`paper`는 2021-10-01~2023-12-31 UTC이고 2023-03-24 13:00 한 봉이 비어 있다. `reused`는 2024-01-01~2026-10-31 UTC다. 연도별 gzip CSV에 UTC 시각, 세 모델의 종가 결정 비중(`decision_*`), 다음 봉 적용 원칙으로 계산한 해당 봉의 수수료·슬리피지·시장충격 차감 순수익률(`net_*`)을 담았다. `student`는 학습 학생, `rule`은 같은 진입 조건의 변동성 비중, `untrained`는 같은 진입 조건의 무학습 학생이다. 연율화 계수는 8,760이며 소르티노 하방편차는 현금 봉을 포함한 전체 관측 수를 사용한다.

## 원 실험 환경의 재현 범위

위 명령은 배포한 순수익에서 지표를 다시 계산한다. 시세부터 비용을 다시 계산하고 학생을 재학습한 기록은 [원 실행 로그](logs/verify_p61_examiner_output.txt)에 있다. 그 실행을 반복하려면 원래 프로젝트 환경과 입력 아카이브가 필요하다. 교사·시세 전체와 모델 가중치는 이 심사 자료에 포함하지 않았다.

- 시세 입력은 `data/cache/btcusdt_1h_20200801_20231231.parquet`, `data/cache/p01_sources/confirmed_ohlcv.parquet`, `data/cache/p05_sources/BTCUSDT_1h.csv`다.
- 교사 입력은 `runs/study2/teacher_signals.npz`와 `consensus_texts.json`이다. FinBERT 로컬 가중치, `runs/baselines/weekly_student/`의 입력 해시 참조 파일과 `runs/baselines/trend_entry/`의 원본 결과·가중치·비교 후보 파일이 추가로 필요하다.
- 원 환경의 읽기 전용 검증 명령은 `python -m ts.baselines.verify_p61_examiner --retrain --sensitivity`다. 해시 불일치 및 누락 입력은 오류로 처리한다. `python -m pytest -q tests src/ts/baselines/tests`가 회귀 검사 명령이다.
- 추세·회복 규칙은 두 기간을 확인하며 60개 이상 후보 규격에서 선택했다. 32개 후보에 한정해 보정한 후속 소르티노 차이(최종 전략 − 기본 추세 전략)의 동시 95% 구간은 [−0.498, 1.992]다. 시드 42 결과와 세 시드 평균을 구분해 해석한다.

## 배포본 검사 기록

공개 저장소의 기준 커밋은 `ec48c3e6c7fb701f65d12770d24c60f8e31e6881`이다. 이 커밋을 별도 디렉터리에 풀고 배포 파일을 적용한 뒤, 기존 BTCUSDT 1시간·1분 시세 캐시를 추가한 환경에서 회귀 테스트 62개가 통과했다. [실행 출력](logs/checkout_tests.txt)

시세 캐시 없이 네트워크도 차단된 상태에서는 전체 회귀 검사 중 시세 로딩을 거치는 검사가 실패할 수 있다. 시간별 성과 검산 명령은 ZIP만 풀어도 오프라인에서 실행된다. 소스 경로와 원 시스템 대장 링크는 저장소 기준이다.
