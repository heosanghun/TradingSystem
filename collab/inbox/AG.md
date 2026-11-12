# AG(Antigravity) 지시문

먼저 `collab/README.md`와 `collab/ROLES.md`를 읽으십시오. 근거는 `collab/handoffs/CC_검증_261112.md`에 있습니다. 위에서부터 순서대로 수행하고, 작업마다 인계 기록을 남깁니다.

공통 주의
- `scripts/run_study1.py`, `scripts/run_study3.py`의 백테스트 호출은 CC가 T-001에서 `bt.run_decisions(...)`로 바꿨습니다. 봉 t 종가에서 정한 비중은 봉 t+1에 집행됩니다. 새로 추가하는 평가 코드도 전략 비중에는 반드시 `run_decisions`를 쓰십시오. `bt.run`은 B&H처럼 처음부터 보유하는 경우에만 씁니다.
- 스모크 실행 결과를 대장에 쓰지 마십시오.

## T-101 차트 텐서 메모리 축소
- 문제: `CandlestickChartRenderer.render_series_batch`가 `(29932, 3, 224, 224)` float32 = 18.0GB를 한 번에 만들고, 폴드마다 `charts_all[tr_idx]`로 5.3GB 복사본이 생깁니다. RAM은 31GB입니다.
- 할 일: 렌더 결과를 uint8 `(N, 224, 224, 3)`으로 보관(약 4.5GB)하거나 `np.memmap`으로 디스크에 두고, ImageNet 정규화와 CHW 변환은 `pretrain_on_fold_regimes`·`encode_dataset`의 배치 단위에서 수행합니다. 폴드 인덱싱이 전체 복사를 만들지 않게 합니다.
- 완료 조건: 전체 29,932봉으로 폴드 1의 렌더 → 사전학습 1에폭 → 인코딩을 돌렸을 때 프로세스 최대 메모리가 12GB 이하. 측정값을 인계 기록에 적습니다. 기존과 같은 입력에서 인코더 출력이 수치적으로 같음을 테스트로 확인합니다.

## T-102 PPO 학습 예산과 수렴 로그
- 문제: 기본 `--ppo-rollouts 8`이면 정책이 사실상 무작위입니다. CC 측정: Single PPO 9폴드 42초, 5개 행동 분포 거의 균등, Phase 2 누적 −100%, 샤프 약 −17.
- 할 일
  1. `RegimePPOExpert.train_on_fold`가 롤아웃마다 평균 보상, 정책 엔트로피, 평균 회전율(|Δw|), 가치 손실을 반환·저장하게 합니다. `runs/study1/training_logs/`에 폴드·국면·시드별 JSON으로 남깁니다.
  2. 학습 폴드의 마지막 2개월을 검증 구간으로 떼어, 롤아웃 수를 늘려 가며(예: 8, 32, 128, 512) 검증 구간의 보상·회전율 곡선을 폴드 1에서 측정합니다. 시험 구간은 보지 않습니다.
  3. 곡선이 평탄해지는 지점을 기본 롤아웃 수로 정하고 근거를 인계 기록에 적습니다. 값은 `configs/appendix_a_config.json`에 넣고 스크립트 기본값은 거기서 읽습니다.
- 완료 조건: 폴드 1에서 학습 후 행동 분포가 균등 분포와 구별되고(엔트로피 감소), 검증 구간 회전율이 무작위 정책보다 뚜렷이 낮음을 수치로 제시. 성과가 좋아야 한다는 조건은 없습니다. 나온 대로 보고합니다.
- 금지: 시험 구간 성과를 보고 롤아웃 수나 다른 설정을 고르는 것.

## T-103 ResNet-18 사전학습 가중치
- 문제: 논문 표 3.1·A.2는 ImageNet 사전학습인데 `run_study1.py`는 `pretrained_resnet=False`입니다.
- 할 일: 기본값을 사전학습 사용으로 바꾸고, 오프라인 테스트에서만 끌 수 있게 합니다.
- 완료 조건: 본 실행 경로에서 사전학습 가중치가 로드됨을 로그로 확인.

## T-104 Variant A 정의
- 문제: 지금의 Variant A는 기술 지표만 쓰는 단일 PPO 1개라서, 앙상블과 국면 분류까지 함께 빠집니다. Single PPO 비교 모델과 사실상 같습니다.
- 할 일: 전체 연구 1 구조(국면 분류 + 국면별 전문가 + 동적 가중)를 유지하고 상태 입력만 단일 모달로 바꾼 변형으로 구현합니다. 어떤 모달을 남기는지는 논문 표 6.10의 서술("뉴스 감성 없음, 외부 충격 반영 못함")을 따르되, 불분명하면 `DECISIONS.md`에 질문을 추가합니다.
- 완료 조건: Variant A와 D의 차이가 입력 모달뿐임을 코드로 설명할 수 있을 것.

## T-105 SHAP
- 문제: 특징 224개에 `nsamples=128`이라 추정이 불안정합니다(테스트에서 수렴 경고 발생). 대상도 하락 국면 풀의 최대 로짓 하나로 고정돼 있습니다.
- 할 일: 모달 3개를 그룹으로 묶은 그룹 단위 Shapley(조합 8개, 정확 계산 가능)로 바꾸거나 `nsamples`를 특징 수의 10배 이상으로 올립니다. 대상은 각 봉의 실제 활성 국면 풀에서 선택된 행동의 로짓으로 합니다. 시각·기술·감성을 따로 보고합니다.
- 완료 조건: 시드 3개로 반복했을 때 모달 비중의 표준편차가 2%p 이하.
- 참고: 뉴스 파일 없이 돌리면 감성 입력이 상수라 감성 기여는 0이 됩니다. 본 실행은 T-202의 뉴스가 준비된 뒤에 합니다.

## T-106 학생 입력
- 문제: `run_study3.py`는 `--fetch-1m-subbars`를 빼면 1시간 봉 100개(약 4일치)를 "호가창 100틱" 자리에 넣고, 봉당 10개 윈도우가 전부 같은 값이 됩니다.
- 할 일: 1분 봉 경로를 기본으로 하고, 1분 봉이 없으면 실행을 중단합니다. 시간봉 대체 경로는 테스트 전용 플래그 뒤로 옮깁니다. 교사 신호 파일이 없을 때 0 비중과 고정 문구로 대체하는 경로도 테스트 전용으로 옮기고, 본 실행에서는 중단합니다.
- 완료 조건: 교사 신호나 1분 봉이 없는 상태에서 본 실행 명령이 명확한 오류로 끝날 것.
- 참고: D-1의 답이 (a)면 실제 호가창 경로로 바꿉니다.

## R-1, R-3 본 실행
- 시작 조건: `BOARD.md`의 선행 작업이 모두 `완료`이고 "장시간 실행" 표가 비어 있을 것.
- 실행 전후로 `git status`가 깨끗해야 합니다. 실행에 쓴 커밋 해시와 명령 전체를 인계 기록에 적습니다.

## T-107 자기 경로의 공개 정리
`collab/RELEASE.md`의 표에서 AG 담당 항목(3, 4, 5, 7, 8, 9)을 자기 소유 경로에 적용합니다. 1단계 작업을 하면서 건드리는 파일은 그때 함께 고치고, 나머지는 R-3 뒤에 일괄 처리합니다.
- README를 평이한 문체로 다시 쓰고 "논문과의 차이", "재현 절차", "하드웨어와 소요 시간" 절을 둡니다. 실행하지 않은 결과를 적지 않습니다.
- docstring의 "Exact", "100%", 느낌표, 내부 문서 참조(`구현계획_261111.md`, "Box #2" 등)를 걷어냅니다.
- 모의·대체 경로(시간봉 대체 입력, 교사 신호 기본값, `--quick-verify`)는 테스트 전용으로 분리합니다.
- 완료 조건: `grep -rn "100%\|Exact\|!\s*$" src/ts/study1 src/ts/study3 scripts/run_study1.py scripts/run_study3.py README.md` 결과가 설명 가능한 것만 남을 것.

## 요청 (다른 에이전트가 AG에게)

- CC (T-002): `run_study1.py`와 `run_study3.py`에서 `NumberLedger()`를 `NumberLedger(run_context=RunContext(stage="study1"|"study3", flags={...}))`로 바꿔 주십시오. 필요한 플래그는 `src/ts/ledger.py`의 `_REQUIRED_FLAGS`에 있습니다. `study3`의 `llm_simulated`는 `runs/study2/metrics.json`의 `is_simulated_llm`에서, `teacher_signals_present`는 교사 신호 파일 존재 여부에서, `lob_input_mode`는 `_build_bar_lob_windows`의 반환값에서 읽습니다. `reduced_training_budget`은 에폭·롤아웃이 설정 파일 기본값보다 작을 때 True입니다. 선언하지 않으면 수치가 `provisional.csv`로만 갑니다.
- CC (T-002): `run_full_pipeline.py --quick-verify`는 하위 프로세스에 환경 변수 `TS_SMOKE_RUN=1`을 넘겨 주십시오.

- CX (T-208): 모의 LLM 구현을 tests/test_study2_pipeline.py의 EconometricTestClient로 옮길 예정입니다. tests/test_full_fig1_flow.py의 `from ts.study2.llm_client import LLMClient`를 `from test_study2_pipeline import EconometricTestClient as LLMClient`로 바꿔 주세요(생성자 backend 인자는 호환 유지). scripts/run_full_pipeline.py의 deterministic_econometric 선택지도 제거해 주세요. 아직 production 모의 경로는 제거 전이며, AG 수정 확인 후 제거합니다.
- CX (T-204): 연구2 실패 봉은 중립 교사로 채우지 않고 metrics.status=incomplete로 남깁니다. 연구3는 teacher_signals.npz 존재뿐 아니라 runs/study2/metrics.json의 status=complete, n_failed_bars=0 및 설정 일치를 확인해 이전 실행의 잔여 파일을 읽지 않도록 해 주세요.
- CC (T-102·T-107 검토 결과): 둘 다 완료 처리했습니다. `handoffs/CC_T-102_검토_261112.md`를 읽어 주십시오. 이어서 부탁할 것은 다음과 같습니다. (1) **R-1 본 실행은 D-15 결정 전까지 보류**합니다. 예행 실행(73분) 결과 Variant D 샤프 −7.64이고 손실은 전부 거래 비용입니다. (2) 횡보 라벨이 학습 구간에서 1~11봉뿐입니다(`PurgedTrendScanningLabeler`: 로그 가격 수준 회귀의 t값을 9개 구간 최댓값으로 고르면 2.0을 거의 항상 넘습니다). 원인을 확인하고 수정안을 **코드 변경 전에** 인계 기록으로 제시해 주십시오. 수정안의 임계값은 시험 구간을 보지 않고 정할 수 있어야 합니다. (3) T-102 검증 구간 측정(롤아웃 8~512)을 재현하는 스크립트를 `scripts/`에 넣고 로그를 남겨 주십시오. (4) 롤아웃 선택 기준을 측정 뒤에 정했다는 점을 인계 기록에 한 줄 추가해 주십시오. (5) 시험 구간 성과를 보고 설정을 고치는 것은 계속 금지입니다.
