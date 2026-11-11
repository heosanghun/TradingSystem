# 숫자 출처 대장 (`ledger/numbers.csv`)

박사학위논문 개정본(`[개정본] 박사논문_개정본_본보기_261110.pdf`) p. 62 **저자 확인** 지침 및 `구현계획_261111.md` 제2절 제2원칙에 따라, 제VI장(표 6.2~6.15)과 본문에 기재되는 모든 실측 수치는 본 디렉토리의 `numbers.csv`에 자동 기록됩니다.

## 컬럼 규격
- `number_id`: 고유 지표 식별자 (예: `T6.8_LUNA_PROPOSED_MDD`, `T6.11_LAMBDA3_0_2_SHARPE`)
- `table_ref`: 논문 내 표/절 위치 (예: `Table 6.8`, `Table 6.11`)
- `metric_name`: 지표 설명
- `value`: 원시 부동소수점 실측치
- `formatted_value`: 논문 표 기입용 포맷 문자열
- `unit`: 단위 (`%`, `ms`, `bars` 등)
- `script_path`: 생성 실행 스크립트 상대 경로
- `config_sha256`: `configs/appendix_a_config.json` SHA-256 해시 앞 16자리
- `git_commit`: 실행 시점 Git 커밋 해시
- `run_log_path`: 원본 JSON 결과 파일 경로 (`runs/.../metrics.json`)
- `recorded_at_utc`: 기록 시각 (UTC ISO-8601)
