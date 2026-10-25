# 플롯 안내

## Tanner CNN: 최신 모델, 두 OSD 방식 함께 표시

[통합 플롯](bb/code_capacity/tanner_cnn/overview.png)은 최신 CNN **68개 설정의 raw 결과**와
그 모델에 대응하는 **새 syndrome OSD-0 25개 결과**를 표시한다.
이전 depth 1–5의 residual OSD 44개 결과도 OSD 패널에 함께 표시한다.
실선·원은 새 syndrome OSD, 점선·사각형은 이전 residual OSD다.

- 위 BB72 / 아래 BB144, 왼쪽 CNN 단독 / 오른쪽 CNN + OSD-0 (두 방식).
- 색은 depth 1–8. 추가 seed는 × 표시다. Raw 패널의 Neural BP4는 다른 학습 조건의 참고선이다.
- 새 OSD 25개 = depth 6/7/8의 native 결과 24개 + BB144 depth 5 p=.04의 전체 사후 평가 1개.
- 최신 depth 1–5 나머지 43개 설정의 전체 새 OSD 결과는 현재 로컬에 없다. 해당 점은
  새 OSD 점은 생략하되, 기존 residual OSD 결과를 별도 점선으로 표시한다.
  서로 다른 OSD 방식의 점들을 하나의 선으로 연결하지 않는다.
- 과거 400-epoch BB144 d5 p=.06 재평가와 256-shot 부분 평가는 최신 800-epoch 모델의
  대체값으로 사용하지 않는다. 사후 평가의 checkpoint/bank/history 해시, config, best epoch,
  전체 shot 수가 일치해야 교체한다.
- 95% Wilson 구간을 표시하고, 실패 0건은 상한 위치에 빈 역삼각형을 표시한다.
- `bb_cnn_latest_final.csv`의 `osd_*` 열은 새 OSD만 포함한다. 미평가 값은 빈 칸이다.
  `recorded_osd_ler`에는 원래 값이 남아 있고 `osd_status`/`osd_source`로 출처를 확인할 수 있다.
  `bb_cnn_all_final.csv`는 과거 단계의 원래 OSD 기록을 보존한다.
  `bb_cnn_residual_osd_overlay.csv`에는 플롯에 추가한 residual OSD 44개 값이 있다.

[표시된 수치](../analysis/bb_cnn_latest_final.csv) ·
[누락된 새 OSD 평가 목록](../analysis/bb_cnn_missing_syndrome_osd.csv) ·
[과거 기록](../analysis/bb_cnn_all_final.csv).

재생성: `python scripts/plot_bb_cnn_all.py`.
새 전체 평가 결과를 `results`의 Tanner CNN archive 또는 `results/analysis`에 추가하면,
같은 명령으로 일치하는 결과를 찾아 반영한다. 이 명령은 추론이나 학습을 실행하지 않는다.

## Depth 6–8 비교

[Depth 비교](bb/code_capacity/tanner_cnn/depth678.png) ·
[Validation 곡선](bb/code_capacity/tanner_cnn/depth678_validation.png) ·
[분석](../analysis/bb_update_2026_10_24.md).

Depth 6–8은 모두 fresh 800 epochs, width 64, 같은 code/p의 동일 final bank다.
Raw 패널의 회색 D5 참조선은 다른 schedule이다. OSD 패널의 회색 점선은
D5의 기존 residual OSD 8개 결과이며, 새 syndrome OSD로 평가한 BB144 p=.04도 별표로
함께 표시한다. Depth 6–8 실선은 새 syndrome OSD이므로 OSD 방식 차이도 고려해야 한다. 재생성: `python scripts/analyze_bb_october24.py`.

이 새 decoder는 zero-free-bit syndrome OSD-0이며 Astra의 전체 확률 변환과 fallback 정책까지
복제한 것은 아니다. 자세한 정의는 [OSD 안내](../../docs/bb_cnn_posthoc_osd.md)에 있다.

## 새 circuit BP/OSD baseline

[BB72 overview](bb/circuit/library_bp_osd/overview.png)는 수정한 0·1번 실험의
p=.001–.008 결과다. 각 지점은 10만 샷이고, BP / BP+OSD-0 / BP+OSD-CS3를
X-only(실선)·joint XZ(점선) 입력에서 비교한다.
`X only`는 X-check detector만 입력한다는 뜻이고, `joint XZ`는 X/Z-check detector를
하나의 DEM/BP 디코더에 함께 넣는다는 뜻이다. 회로의 오류 모형은 같으며, 두 방법 모두
logical-X 관측값 12개의 flip, 즉 logical-Z 오류를 예측한다. X-only가 물리 X 오류만
발생시키는 실험이라는 뜻은 아니다. Circuit detector는 반복 측정 사이의 check 변화 정보다.
왼쪽은 **logical-Z 관측값 불일치율**, 오른쪽은 **syndrome 불일치까지 포함한 block failure**다.
두 패널은 같은 샷의 서로 다른 지표이며 OSD의 값은 두 패널에서 같다.
Error bar는 95% Wilson 구간이다. p=.008은 최종 집계가 완료됐지만 로컬 chunk 다운로드가 일부 누락됐다.
[수치·검증·해석](../analysis/bb_library_bp_osd_2026_09_30.md)을 함께 참고한다.

**같은 회로를 쓴 Relay BP·Neural Relay BP·Neural BP·Neural BP+OSD는
[neural_bp/overview.png](bb/circuit/neural_bp/overview.png) 한 장에서 본다.**
왼쪽 BB72 / 오른쪽 BB144이며, 네 곡선 모두 idle 오류 없는 회로의 X/Z block decoding failure다.
수정한 일반 BP / BP+OSD는 noise와 평가 대상이 달라 `library_bp_osd/`에 별도로 둔다.

Relay BP와 Neural Relay BP는 코드별 p=.001/.002/.003/.004/.005/.006/.008/.010의 8개 점이다.
Neural BP는 코드별 p=.001–.004의 4개 점, Neural BP+OSD는 BB72 8개 / BB144 4개 점이다.
모두 완료된 4,096-shot 최종 평가이며, 곡선별 p-grid를 보존했다. 없는 값을 보간하거나
OSD-selected checkpoint의 raw 결과로 대체하지 않는다. Error bar는 95% Wilson 구간이고,
실패 0회는 로그 축에서 0.5/N 위치에 아래쪽 삼각형으로 표시한다. 이 위치가 측정 LER은 아니다.

Relay는 최대 4 legs × 12 iterations와 2-solution search, 일반 Neural BP는 12 iterations다.
따라서 같은 연산량 비교는 아니다. Neural BP의 raw/OSD checkpoint는 따로 선택됐고,
Neural+OSD는 원래 posterior-seeded wrapper 결과다. Relay/Neural Relay만 같은 샷의 paired 비교다.

기존 `raw_vs_osd.png`와 Relay 전용 overview·training PNG 3장은 제거했다.
학습·validation 기록, paired gain·유의성 검정, ablation 및 원본 실험 데이터는
`results/analysis/`와 `results/bb/circuit/`에 보존했다. 재생성해도 전용 Relay 폴더를 만들지 않는다.

이 그림은 X/Z 양쪽 논리 관측값 24개의 block failure를 평가한다. 새 library baseline은
idle noise가 있는 회로에서 logical-Z 성분 12개를 평가하므로 두 그림의 LER 차이를
같은 decoding 문제의 성능 차이로 읽으면 안 된다. 새 BB144 library 결과는 아직 없다.

## 폴더 기준

날짜 대신 **코드 → noise → 모델**로 분류한다. 날짜와 job ID는 분석 보고서 및 원본 결과 폴더에 남긴다.

```text
results/plots/
├── README.md                         ← 이 안내
├── bb/
│   ├── code_capacity/                ← 데이터 오류 + 완벽한 syndrome
│   │   ├── tanner_cnn/               ← 현재 joint Tanner CNN
│   │   │   └── overview.png          ← 설정별 최신 CNN 결과
│   │   └── neural_bp/                ← 이전 Neural BP4 / classical 비교
│   │       ├── overview.png
│   │       └── ablations.png
│   └── circuit/                      ← 회로·측정 오류
│       ├── library_bp_osd/           ← 수정한 BP / BP+OSD baseline
│       │   └── overview.png
│       └── neural_bp/                ← Relay / Neural Relay / Neural / Neural+OSD
│           └── overview.png
└── toric/
    └── phenomenological/
        └── threshold.png
```

## 이전 실험 찾기

| 실험 | 결과 그림 | 보조 그림 / 해석 |
| --- | --- | --- |
| BB capacity Neural BP4 | [overview](bb/code_capacity/neural_bp/overview.png) | [구성요소 비교](bb/code_capacity/neural_bp/ablations.png), [보고서](../analysis/bb_campaign_2026_08.md) |
| BB circuit Relay / Neural BP | [네 방법 통합 비교](bb/circuit/neural_bp/overview.png) | [Neural BP 보고서](../analysis/bb_circuit_campaign_2026_08.md), [Relay 분석·학습 기록](../analysis/bb_neural_relay_bb144_2026_09_15.md) |
| Toric ConvGRU / PyMatching | [threshold](toric/phenomenological/threshold.png) | [선택한 데이터](../analysis/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv) |

새 논문 비교용 plain BP/BP+OSD baseline은 [현재 0–4번 실험](../../docs/bb_baseline_campaign.md)이다.
현재 BB72 baseline은 별도 library 그림에서만 표시한다.
Code capacity와 circuit의 p는 서로 다른 오류 모형의 확률이므로 곡선을 직접 비교하지 않는다.
이전 circuit의 raw/OSD 그림은 서로 다른 기준으로 고른 checkpoint의 비교다.

## Block LER 읽는 법

BB72와 BB144는 둘 다 **논리 큐빗 12개**를 담는다. 한 번의 오류 샘플과 디코딩을 한 블록의
평가로 보고, 어느 논리 큐빗에서든 logical X/Z 오류가 남으면 그 샷을 실패 1회로 센다.
논리 오류가 여러 개 남아도 실패 샷 수는 1만 증가한다.

현재 Tanner CNN의 `Block LER`는 다음의 **block decoding failure rate**다.

```text
Block LER = (syndrome 불일치 샷 수
             + syndrome은 맞지만 logical 오류가 남은 샷 수) / 전체 샷 수
```

예를 들어 1,000샷 중 실패한 블록이 50개면 Block LER=0.05=5%다.
물리 큐빗별 오류율이나 논리 큐빗 하나당 오류율이 아니며, 12로 나눈 값도 아니다.
실제 오류와 correction이 달라도 둘의 차이가 stabilizer이면 성공으로 센다.
CNN+OSD는 모든 syndrome을 맞추므로 남은 실패가 모두 logical 오류다.

새 circuit baseline은 syndrome 불일치까지 세는 `block_failure_rate`와, 논리 관측값만 비교하는
`logical_z_mismatch_rate`를 따로 저장한다. 논문과 비교할 때 어떤 정의인지 확인한다.

Library 그림의 **Logical-Z prediction error**는 예측한 logical-X 관측값 flip 벡터
12비트가 실제 벡터와 하나라도 다른 샷의 비율이다. Z 성분 오류가 logical-X 측정을
뒤집으므로 이름이 logical-Z다. 12비트 평균 오류율이나 물리 Z 오류율이 아니며,
6개 noisy round 전체 memory experiment에 대한 block 단위 값이다.
예를 들어 1,000샷 중 10샷에서 한 개 이상의 비트 예측이 틀리면 1%다.
추정 correction의 syndrome 불일치 자체는 이 왼쪽 지표에서 추가 실패로 세지 않는다.

## 재생성

아래 명령은 저장된 결과로 그림을 다시 만든다. 학습이나 디코더 재실행이 아니다.

```bash
# 새 circuit library BP / BP+OSD: overview (최종 집계 검증 후 그림만 갱신)
python scripts/summarize_bb_library_baselines.py --plots-only
# --plots-only를 빼면 저장 correction 재채점과 전체 audit도 수행

# 현재 Tanner CNN: overview / training / seeds
python scripts/summarize_bb_september27.py

# 이전 capacity Neural BP4: overview / ablations
python scripts/summarize_bb_campaign.py

# circuit Relay BP / Neural Relay BP / Neural BP / Neural BP+OSD: 통합 overview 한 장
python scripts/plot_bb_circuit_raw_vs_osd.py
# 기존 파일명은 유지했지만 출력은 neural_bp/overview.png이다.
# Relay 스크립트로도 같은 그림을 갱신할 수 있다:
# python scripts/summarize_bb_neural_relay.py --plots-only

# Toric: 선택된 threshold 데이터만 사용
python scripts/plot_threshold.py \
  results/analysis/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv \
  --out results/plots/toric/phenomenological/threshold.png \
  --title "Phenomenological Threshold: ConvGRU vs PyMatching"
```

Toric 재생성에는 위의 선택된 CSV를 사용한다. 전체 archive를 넘기면 같은 `(L,p)`의
서로 다른 learning-rate branch를 반복 실험으로 취급해 평균낼 수 있다.

Circuit Relay/Neural BP 그림은 `neural_bp/overview.png` 한 장만 유지한다.
PDF와 중복 복사본은 만들지 않았다.
스크립트 기본 저장 경로도 이 구조를 따른다.
