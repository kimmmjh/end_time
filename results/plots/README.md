# 플롯 안내

**지금 진행 중인 joint Tanner CNN은 [성능 비교](bb/code_capacity/tanner_cnn/overview.png)부터 보면 된다.**

| 순서 | 현재 Tanner CNN 그림 | 보는 내용 |
| --- | --- | --- |
| 1 | [overview.png](bb/code_capacity/tanner_cnn/overview.png) | 왼쪽 BB72 / 오른쪽 BB144. 각 패널에 CNN depth 1·2의 raw/OSD 4개 곡선과 OSD 없는 Neural BP4를 함께 표시 |
| 2 | [training.png](bb/code_capacity/tanner_cnn/training.png) | 20개 모델의 학습 loss와 validation LER. 실선 raw, 점선 OSD, 다이아몬드 별도 final test |
| 3 | [seeds.png](bb/code_capacity/tanner_cnn/seeds.png) | p=.06, depth 2, 코드별 3개 seed의 raw/OSD LER |

모두 **code-capacity depolarizing noise** 결과이며 circuit-level 결과가 아니다.
Neural BP4의 학습 샘플은 CNN의 12배이며 test bank도 다르다. 주 그림 하단에 이 차이를 표시했다.
Neural BP4는 공통 p=.04/.06/.08에만 그린다. 두 패널의 y축 범위는 다르며, error bar는 shot 통계의 95% Wilson 구간이다.
상세 수치와 조건은 [Tanner CNN 분석](../analysis/bb_update_2026_09_27.md)에 있다.

## 폴더 기준

날짜 대신 **코드 → noise → 모델**로 분류한다. 날짜와 job ID는 분석 보고서 및 원본 결과 폴더에 남긴다.

```text
results/plots/
├── README.md                         ← 이 안내
├── bb/
│   ├── code_capacity/                ← 데이터 오류 + 완벽한 syndrome
│   │   ├── tanner_cnn/               ← 현재 joint Tanner CNN
│   │   │   ├── overview.png
│   │   │   ├── training.png
│   │   │   └── seeds.png
│   │   └── neural_bp/                ← 이전 Neural BP4 / classical 비교
│   │       ├── overview.png
│   │       └── ablations.png
│   └── circuit/                      ← 회로·측정 오류를 포함한 이전 실험
│       ├── neural_bp/
│       │   ├── overview.png
│       │   ├── ablations.png
│       │   └── raw_vs_osd.png
│       └── neural_relay/
│           ├── overview.png
│           ├── training_bb72.png
│           └── training_bb144.png
└── toric/
    └── phenomenological/
        └── threshold.png
```

## 이전 실험 찾기

| 실험 | 결과 그림 | 보조 그림 / 해석 |
| --- | --- | --- |
| BB capacity Neural BP4 | [overview](bb/code_capacity/neural_bp/overview.png) | [구성요소 비교](bb/code_capacity/neural_bp/ablations.png), [보고서](../analysis/bb_campaign_2026_08.md) |
| BB circuit Neural BP2 | [overview](bb/circuit/neural_bp/overview.png) | [구성요소 비교](bb/circuit/neural_bp/ablations.png), [raw/OSD 비교](bb/circuit/neural_bp/raw_vs_osd.png), [보고서](../analysis/bb_circuit_campaign_2026_08.md) |
| BB circuit Neural Relay BP | [overview](bb/circuit/neural_relay/overview.png) | [BB72 학습](bb/circuit/neural_relay/training_bb72.png), [BB144 학습](bb/circuit/neural_relay/training_bb144.png), [보고서](../analysis/bb_neural_relay_bb144_2026_09_15.md) |
| Toric ConvGRU / PyMatching | [threshold](toric/phenomenological/threshold.png) | [선택한 데이터](../analysis/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv) |

이전 circuit 그림은 당시의 noise·decoder 설정 결과다. 새 논문 비교용 plain BP/BP+OSD baseline은
[현재 0–4번 실험](../../docs/bb_baseline_campaign.md)이며, 그 결과 플롯은 아직 없다.
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

## 재생성

아래 명령은 저장된 결과로 그림을 다시 만든다. 학습이나 디코더 재실행이 아니다.

```bash
# 현재 Tanner CNN: overview / training / seeds
python scripts/summarize_bb_september27.py

# 이전 capacity Neural BP4: overview / ablations
python scripts/summarize_bb_campaign.py

# 이전 circuit Neural BP2: overview / ablations / raw_vs_osd
python scripts/summarize_bb_circuit_campaign.py
python scripts/plot_bb_circuit_raw_vs_osd.py

# 이전 circuit Neural Relay BP: overview / training_bb72 / training_bb144
python scripts/summarize_bb_neural_relay.py --code all

# Toric: 선택된 threshold 데이터만 사용
python scripts/plot_threshold.py \
  results/analysis/threshold_ConvGRU_PyMatching_L9_L11_L13_L15.csv \
  --out results/plots/toric/phenomenological/threshold.png \
  --title "Phenomenological Threshold: ConvGRU vs PyMatching"
```

Toric 재생성에는 위의 선택된 CSV를 사용한다. 전체 archive를 넘기면 같은 `(L,p)`의
서로 다른 learning-rate branch를 반복 실험으로 취급해 평균낼 수 있다.

폴더 정리 때 기존 PNG 12개를 보존했고, 이후 Tanner CNN overview는 Neural BP4와 직접 비교하도록 갱신했다.
PDF와 중복 복사본은 만들지 않았다.
스크립트 기본 저장 경로도 이 구조를 따른다.
