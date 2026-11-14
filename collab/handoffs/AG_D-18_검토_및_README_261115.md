# AG 인계: D-18·단일 모델 t-SNE 변경 코드 검토 및 README 반영 — 2026-11-15 08:54

- 담당: AG (Antigravity)
- 상태: 진행 (2회차 무인 연쇄 실행 완료 후 최종 소요 시간 갱신 및 단위 테스트 확인 예정)
- 대상 커밋: `eb8dc10`, `afd05d9` (`src/ts/study3/student_model.py`, `src/ts/config.py`, `scripts/run_study3.py`, `scripts/calibrate_study3_action_gate.py`)

## 1. 변경 코드 검토 결과 (이슈 없음)

1. **인과성 및 폴드 경계 연속성 (`apply_action_confidence_gate`, `src/ts/study3/student_model.py`, `scripts/run_study3.py`)**:
   - `apply_action_confidence_gate(probs, threshold, initial_action=2)`는 `t = 0 .. N-1` 순방향으로만 순회하며 초기값 `action = 2`(`w = 0.0`, 현금)에서 시작해 `probs[t, top] >= threshold`일 때만 행동을 전환하고, `threshold <= 0`일 때는 기존 `np.argmax(probs, axis=1)`과 항등이다.
   - Phase 2(`res_p2`, `table_6_11`, `agreement_pct`)에서는 9개 Walk-Forward 시험 폴드의 확률 행렬 `oos_probs_by_l3[l3]`을 시간순으로 먼저 이어 붙인 뒤 게이트를 1회 적용하므로 폴드 경계에서도 보유 행동이 끊기지 않고 실제 연속 운용과 일치한다.
   - Phase 1(`p1_pred_weights`)에서도 5-폴드 교차검증 예측 확률을 `idx_to_local`로 평가 구간 시간순(`cat_te_idx`)에 재정렬한 뒤 동일한 `apply_action_confidence_gate(p1_probs, action_theta)`와 `bt.run_decisions(..., phase=1)`를 거치므로 미래 참조가 없다.
2. **단일 학생 잠재 공간 인코딩 범위 (`LATENT_FIGURE_FOLD_ID = 2`, `scripts/run_study3.py`)**:
   - `Fold 2`의 학습 구간은 `2021-01-01`~`2021-12-31`이고 `te_idx[0]`은 `2022-01-01 00:00 UTC`이므로, `fixed_idx = np.arange(int(te_idx[0]), len(df_all))` 구간의 모든 봉(`2022-01`~`2023-12`)은 `Fold 2` 학생 모델이 학습 중 한 번도 보지 않은 순수 미관측 구간임을 확인했다.
   - 기존 `runs/study3/student_oos_hidden.npz` 저장 경로도 그대로 유지되어 `tests/test_study3_distillation.py`의 `--hidden-file` 왕복 테스트와 충돌하지 않는다.
3. **설정 해시 보존 (`Study3Config.action_confidence_threshold = 0.8`, `src/ts/config.py`)**:
   - `configs/appendix_a_config.json`(`config_sha256 = d16038cd1c2cbf63`)을 변경하지 않고 `Study3Config` 기본값으로 두어 `R-2` 완료 산출물(`runs/study2/metrics.json`)과의 해시 정합성이 그대로 유지된다.

## 2. `README.md` 반영 내역 (GPU 미사용)

1. **1~2절**: 온라인 추론 경로의 `apply_action_confidence_gate` 집행 규칙 및 `tools/run_pipeline_after_r2.ps1` 디렉토리 구성을 반영했다.
2. **3절 (논문과의 차이, 총 12항목)**: `RELEASE.md` 점검표(내부 작업 기호 제거 및 `DECISIONS.md` 전 항목 정합)에 맞춰 (6항) 연구 2 토론 주기(`--debate-stride 4`)·반사실 표본(`--counterfactual-every 2`)·일간 엣지 상한(`--edge-top-k 30`) 및 부록 B 원문 검증기 승인율(`3.42%`, `1,217 / 35,610`), (9항) 소거·민감도 평가 단계 구분, (11항) 연구 3 행동 전환 신뢰도 게이트(`0.8`, 1회차 시험 구간 원인 분해 뒤에 도입되고 수치만 폴드 1 내부 두 분할 검증으로 동결된 순서 및 내부 검증 `96~99.7%` 현금 유지 사실 명시)와 `Fold 2` 단일 학생 기반 잠재 공간 t-SNE(`student_single_model_hidden.npz`)를 평이한 문체로 반영했다.
3. **5절 (하드웨어와 소요 시간)**: `R-2` 실측(50봉 캐시 우회 교사 지연 평균 `16,835.8 ms`·중앙값 `11,027.4 ms`·P99 `36,758.9 ms`, 28,468봉 완주 경과 약 `53시간`) 및 1회차 `R-3`(`2시간 10분`)·`R-4`(`28분 26초`)·행동 게이트 내부 검증(약 `3분`) 실측 시간을 반영했다.
4. **문체 정규식 점검**: `README.md`, `src/ts/study1/*.py`, `src/ts/study3/*.py`, `src/ts/future_work/*.py`, `scripts/run_study1.py`, `scripts/run_study3.py`, `scripts/run_full_pipeline.py`, `scripts/calibrate_*.py`에 대해 `100%|Exact|!\s*$` 검출 `0`건을 확인했다.

## 3. 후속 예정

- `C:/Project/TradingSystem_final/runs/pipeline.log`의 2회차 무인 연쇄 실행(`afd05d9`)이 `pipeline complete`로 끝나 GPU·파일 수정 잠금이 해제되면, 2회차 `R-3`·`R-4` 실측 소요 시간을 `README.md` 5절에 최종 반영하고 `pytest -q` 전체 통과를 확인하여 `T-107` 2차를 마무리한다.
