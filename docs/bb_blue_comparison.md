# Blue 논문과 새 circuit baseline

[Blue et al., arXiv:2504.13043v2](https://arxiv.org/html/2504.13043v2)의
Fig. 1/3 LER 비교군은 **BP-OSD-3**이며 plain BP 단독 곡선은 없다.

새 실험은 논문의 logical-plus memory task, noisy idle, logical Z 오류 지표를
따른다. X-check-only 입력과 joint X/Z 입력을 같은 회로 샘플에서 평가한다.
두 입력 모두 BP, BP+OSD-0, BP+OSD-CS3를 저장한다.

BP 최대 1000회·min-sum 계수 1.0·OSD-CS3는
[Diffusion §IV.6](https://arxiv.org/html/2509.22347v1)에 명시된 설정이다.
Blue 본문에는 BP 최대 반복 수가 명시되어 있지 않으며 Diffusion은 CUDA-Q를
사용한다. 우리 구현은 `ldpc==2.4.1`이므로 모든 논문의 구현과 완전히 같다고
주장하지 않는다. 회로 순서는 원 논문 저자 공개 코드에 맞췄으며, 수치 재현은
새로운 통계가 나온 뒤 평가해야 한다.

| 항목 | 새 baseline |
| --- | --- |
| noisy cycles | BB72 6회, BB144 12회 + 완전한 reference/closing |
| 측정 스케줄 | 저자 공개 코드의 8 tick, native RX/MX/R/M |
| noise | 초기화·측정 p, idle DEP1(p), CNOT DEP2(p) |
| plain BP | 라이브러리 min-sum, 계수 1.0, 최대 1000회, 수렴 시 종료 |
| OSD | 같은 BP 실행의 OSD-0 및 combination sweep order 3 |
| 논문 비교 지표 | 전체 실험의 12-bit logical Z mismatch rate |
| 추가 지표 | syndrome 실패를 포함한 correction block failure rate |

이전 `idle=0`, 24 logical observables, 계수 0.625 결과는 이 baseline으로
사용하지 않는다. 해당 5개 sweep 데이터는 사용자 요청으로 삭제했다.
실험 구성·통계 예산·재개 방법은 [실행 문서](bb_baseline_campaign.md)에 있다.
