# Result archive

## 2026-11-05 완료 기록 교체 — 현재 상태

[최신 분석](analysis/bb_dem_2026_11_05.md) · [Raw + hierarchical 통합 그림](plots/bb/circuit/standard/dem_cnn/overview.png)

`60701325_followup_train`과 `60701066_followup_evaluate`의 완료 사본(94개 파일)을 검증해
`bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`의 중간 사본을 교체했다.
이전 사본과 루트 중복 폴더는 제거했다. 이제 후속 학습 21개, 평가 22개 모두 완료됐다.
200만 샘플 실험 결과는 아직 없다.

BB144 D8 raw는 p=0.001/0.002에서 각각 35.950→34.106%, 78.894→74.329%다.
반면 BB72 D12 p=0.002 raw는 28.918→31.482%로 높아졌다.
BB144 D4 p=0.004 hierarchical은 best(epoch 5) 92.859% → latest(epoch 800) 89.111%다.

통합 그림 한 장에 raw 32개 점(1,000-epoch 예산 21개 + 800-epoch 예산 11개),
hierarchical 27개 점을 표시한다. Raw는 최신 완료 run의 validation-best,
hierarchical은 표시된 평가 샷에서 최저 LER인 checkpoint다. 후자는 독립 최종 평가가 아니다.
재생성: `python scripts/analyze_bb_dem_november5.py`.
아래 날짜별 상태와 CSV는 당시 수신 기록으로 보존한다. 11월 3일/4일 스크립트를 실행하면
완료 사본 교체를 감지해 최신 보고서와 통합 그림을 재생성한다.

## 2026-11-04 추가 결과 수신

[최신 분석](analysis/bb_dem_2026_11_04.md) ·
[Raw + hierarchical 통합 그림](plots/bb/circuit/standard/dem_cnn/overview.png).

새 6개 폴더 / 243개 파일을 SHA-256 검증 후
`bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`에 보관했다.
새 완료는 학습 5개와 평가 14개다. 누적 학습 21개 중 17개 완료/4개 중간,
평가 22개 중 18개 완료/4개 중간이며 미수신 worker는 없다.
중간 기록은 이전 다운로드 상태다. 200만 샘플 실험 결과는 아직 없다.

BB144 p=0.4%에서 hierarchical 실패율은 best(epoch 5)→latest(epoch 800) 비교 시
D6 94.727→84.131%, D8 94.360→68.835%다. 이는 checkpoint 선택 효과이며
800→1,000 epoch 추가 학습 효과와 구분한다. 새 BB144 D12 raw는 p=0.1%/0.2%에서
각각 15.735→13.623%, 58.704→56.592%다. 새 `.pt`/`.npz`는 없어 paired 검정은 하지 않았다.

현재 공용 플롯은 raw/hierarchical을 함께 담은 **통합 그림 한 장만** 유지한다. Raw는 최신 완료 run의 raw-validation best만,
hierarchical은 완료 평가 중 configuration별 최저 관측 LER만 표시한다. Hierarchical은 표시된 평가 샷으로
선택했으므로 독립 최종 성능 추정치가 아니다. Raw는 1,000-epoch 예산 17개 + 800-epoch 예산 15개다.
재생성: `python scripts/analyze_bb_dem_november4.py`.
아래 날짜별 내용은 당시 기록이며 공용 plot 경로는 최신 그림을 가리킨다.


## 2026-11-03 추가 학습 / best–latest 평가 수신

[분석 보고서](analysis/bb_dem_2026_11_03.md) ·
추가 학습 전후 (historical plot removed; numerical results retained) ·
Best/latest hierarchical 비교 (historical plot removed; numerical results retained).

새 6개 폴더(280개 파일)를 `bb/circuit/standard/dem_cnn/followup_2026_11_02/{train,evaluate}/`에
SHA-256 검증 후 정리했다. 추가 학습 12개는 1,000 epoch 완료, 4개는 중간 snapshot이다.
Hierarchical은 4개 checkpoint 평가 완료, 4개 중간 집계다. 미완료는 다운로드 시점 상태이며
서버의 현재 상태를 의미하지 않는다. `.pt`/`.npz`는 없어 history/metadata/log만 검증했다.

BB72 D8 p=0.2%의 raw 실패율은 67.834→63.477%, BB144 D6 p=0.2%는 87.146→84.058%다.
Best epoch가 그대로인 3개 모델은 재평가 차이를 학습 효과로 해석하지 않는다.
BB144 D4 p=0.4% hierarchical은 공통 6,016샷의 **중간 집계**에서 best(epoch5) 92.886%,
latest(epoch800) 89.195%다. 전체 평가와 실제 paired 샷 검증은 아직 필요하다.

기존 raw overview, validation/loss 및 hierarchical depth/비용 플롯을 갱신했다.
현재 raw overview는 1,000 epoch 12개 + 800 epoch 20개이므로 동일 학습 예산 비교가 아니다.
새 결과로 학습이나 추론을 실행하지 않았다. 재생성: `python scripts/analyze_bb_dem_november3.py`.
아래 날짜별 내용은 과거 분석이며 공용 plot 경로는 최신 그림을 가리킨다.

## November 2, 2026: full hierarchical campaign and BB144 D12 final

Six new job folders (220 files, 5,095,594 bytes) were archived with SHA-256
verification. The five hierarchical jobs join the prior BB72 D4 job under
`bb/circuit/standard/dem_cnn/hierarchical/{bb72,bb144}/`: 24 evaluated models,
196,608 total shots. BB144 D12 job `60636624` is under
`bb/circuit/standard/dem_cnn/resumed_800_epochs/bb144/`; all four models now
have 800-epoch final results, bringing the raw training campaign to 32 models.

[Full analysis](analysis/bb_dem_2026_11_02.md) ·
Hierarchical depth comparison (historical plot removed; numerical results retained) ·
Decoder use and timing (historical plot removed; numerical results retained) ·
[Updated raw overview](plots/bb/circuit/standard/dem_cnn/overview.png).

BB144 D4 at p=0.1% has zero hierarchical failures in 8,192 shots (95% Wilson
upper bound 0.0469%, not proof of zero error). Direct CNN+OSD was not compared.
Imports again exclude `.npz`/`.pt`, so audit coverage is metadata and logs.
Rebuild current plots: `python scripts/analyze_bb_dem_november2.py`.
Older dated summaries below describe their original import state.

## November 1, 2026: BB72 D4 hierarchical DEM evaluation

Job `60636229` (array task 0) is archived under
`bb/circuit/standard/dem_cnn/hierarchical/bb72/resdir_60636229_hierarchical/`.
All 37 files (800,994 bytes) were verified by SHA-256 before/after relocation.
Four p values each completed 8,192 fresh paired circuit shots using existing
best weights, confidence 0.9 and residual OSD-0, with no retraining.

At p=0.1%/0.2%, raw block failure falls from 16.772%/54.028% to
0.110%/3.174%. All hierarchical outputs satisfy the syndrome; high-p logical
failure remains large. Direct CNN+OSD comparison was disabled, and no shot
banks/checkpoints were downloaded, so the audit covers metadata and logs only.

[Analysis](analysis/bb_hierarchical_dem_2026_11_01.md) ·
[Performance](plots/bb/circuit/standard/dem_cnn/overview.png) ·
Diagnostics (historical plot removed; numerical results retained).
Rebuild: `python scripts/analyze_bb_hierarchical_dem.py`.

## November 1, 2026: depth 1 transfer folders returned to archive

Jobs `58793764` (BB72) and `58793765` (BB144), Tanner CNN depth 1 / 100 epochs,
are back under `bb/code_capacity/depolarizing/tanner_cnn/{bb72,bb144}/`.
All 100 files (40,297,607 bytes) were verified by SHA-256 before and after relocation.
The October 12 root locations below are historical; existing discovery supports the archive.
[Relocation manifest](analysis/bb_root_cleanup_2026_11_01.csv).

## November 1, 2026: BB144 D8 completion and D12 resumed timeout

[최신 분석](analysis/bb_dem_2026_11_01.md) · [최신 플롯](plots/bb/circuit/standard/dem_cnn/overview.png).

새 100개 파일의 SHA-256을 검증해 정리했다. 원래 D8 `60318790` 및 재중단 D12
`60428753`은 `bb/circuit/standard/dem_cnn/interrupted/bb144/`, 완료 D8
`60428756`은 `bb/circuit/standard/dem_cnn/resumed_800_epochs/bb144/`에 있다.
D8 네 모델은 총 800 epochs 완료, D12는 790/798/787/795 epochs (p 오름차순), final 없음.
Overview는 28개 완료 모델, validation/loss는 최신 D12 중간 기록도 포함한다.
재생성: `python scripts/analyze_bb_dem_november1.py`. 아래 날짜별 항목은 당시 기록이다.

## October 28, 2026: BB144 depth 12 timeout record

[중단 결과 분석](analysis/bb_dem_2026_10_28_interrupted.md) ·
학습 곡선 (historical plot removed; numerical results retained).

`resdir_60310800` (array 60310800/7)은 24시간 제한으로 중단된 원래 D12 학습이다.
34개 파일 / 1,375,378 bytes를 해시 검증하여
`bb/circuit/standard/dem_cnn/interrupted/bb144/resdir_60310800/`로 정리했다.
p=.001/.002/.004/.006은 각각 394/399/393/398 epochs까지 기록됐고 final 평가는 없다.
Overview는 완료된 24개 모델만 표시하고, validation/loss에는 중단된 D12 곡선을 점선으로 추가했다.
BB144 D8은 여전히 미수신이며 D12 재개 학습 결과도 아직 없다.
재생성: `python scripts/analyze_bb_dem_interrupted.py`.

## October 28, 2026: standard circuit-level DEM CNN

[분석](analysis/bb_dem_2026_10_28.md) ·
[성능 그림](plots/bb/circuit/standard/dem_cnn/overview.png) ·
Validation (historical plot removed; numerical results retained) ·
Logical parity loss (historical plot removed; numerical results retained).

Jobs 60310802/60310803/60316098/60316182/60318763/60330386의 24개 모델을
`bb/circuit/standard/dem_cnn/800_epochs/{bb72,bb144}/resdir_JOB/`로 정리했다.
252개 파일 / 15,356,901 bytes의 이동 전후 SHA-256을 검증했다.
BB72 D4/6/8/12, BB144 D4/6 × p=.001/.002/.004/.006이며 모두 800 epochs 완료.
BB144 D8/12의 final 결과는 없다(D12 중단 기록은 위 항목 참고). 전부 **OSD 없는 raw CNN** 평가다.

모델당 training 409,600 shots, final 8,192 circuit shots.
History와 train/validation/final 로그, 종료 코드 및 66개 source snapshot을 검증했다.
가중치·shot bank는 다운로드에 없어 추론 재현과 paired depth 검정은 하지 않았다.
BB72 D12는 낮은 p에서 개선되지만 높은 p의 block failure는 여전히 높다.
BB144 p=.004/.006은 모든 validation 성공이 0건이라 epoch 5 checkpoint가 선택됐다.
Block failure, observable mismatch, syndrome consistency를 별도 지표로 기록했다.
재생성: `python scripts/analyze_bb_dem_october28.py`.

## October 26, 2026 evening: depth 10/12 and residual OSD complete

[분석](analysis/bb_update_2026_10_26_evening.md) ·
[전체 플롯](plots/bb/code_capacity/tanner_cnn/overview.png) ·
[D6/7/8/10/12 동일 residual 비교](plots/bb/code_capacity/tanner_cnn/depth_comparison.png) ·
[학습 곡선](plots/bb/code_capacity/tanner_cnn/depth1012_validation.png).

새 4개 폴더 / 165개 파일 / 8,309,136 bytes를 이동 전후 SHA-256 검증하여 정리했다.
Jobs 60255257(BB72 D10), 60261485(BB144 D10), 60261486(BB72 D12)는
`bb/code_capacity/depolarizing/tanner_cnn/800_epochs/residual_osd0/{bb72,bb144}/`에 보관했다.
새 12개 모델 모두 800 epochs 완료; 기존 BB144 D12를 합쳐 D10/12 16개가 모두 수신됐다.
Job 60255150의 D6/7/8 재평가 24개는 같은 Tanner CNN 경로의
`posthoc_osd0/depth678/resdir_60255150/`에 보관했다.
이는 code-capacity 결과이며 DEM circuit-level 결과는 아니다.

D10/12는 history/log 검증이며 체크포인트·샷 파일은 없다. D6/7/8 재평가는
로컬 원본 checkpoint/bank/history 해시를 보고서와 대조했다. 재평가 paired 샷 파일은
없으므로 paired count와 residual metric은 서버 보고서 기준임을 기록했다.
최신 전체 플롯은 raw 84개와 residual OSD 84개를 표시한다. Syndrome OSD 25개는 자료만 보존한다.
재생성: `python scripts/analyze_bb_october26_evening.py` 및 `python scripts/plot_bb_cnn_all.py`.

## October 26, 2026: BB144 depth 12

[분석](analysis/bb_update_2026_10_26.md) · [전체 플롯](plots/bb/code_capacity/tanner_cnn/overview.png) · [D6/7/8/12 비교](plots/bb/code_capacity/tanner_cnn/depth12.png).

Job 60255255 (array task 3)의 네 p 결과를
`bb/code_capacity/depolarizing/tanner_cnn/800_epochs/residual_osd0/bb144/resdir_60255255/`에 보관했다.
39개 파일 / 2,691,924 bytes의 이동 전후 SHA-256을 확인했다. 모두 800 epochs 완료.
체크포인트·샷 파일은 다운로드에 없어 history/log 검증만 수행했으며 paired 검정은 하지 않았다.
이 첫 다운로드 당시 미수신이었던 나머지 결과는 위 저녁 업데이트에서 추가했다.
이 시점 분석 재생성: `python scripts/analyze_bb_october26.py`.

## October 24, 2026: depth 6–8 and depth 5 completion

[결과 분석](analysis/bb_update_2026_10_24.md) ·
[성능 그림](plots/bb/code_capacity/tanner_cnn/depth678.png) ·
[학습 곡선](plots/bb/code_capacity/tanner_cnn/depth678_validation.png).

새 7개 job / 25개 모델 모두 800 epochs 완료. Depth 6–8은
`bb/code_capacity/depolarizing/tanner_cnn/800_epochs/syndrome_osd0/{bb72,bb144}/`,
BB144 depth 5 p=.04의 job 59707467은 같은 Tanner CNN 경로의
`continued_training/`에 보관했다. 338 files / 869,903,041 bytes의 이동 전후
해시를 검증했다. [Manifest](analysis/bb_update_2026_10_24_relocation.csv).

Depth 6→8은 raw 6/8조건, 새 syndrome OSD-0 4/8조건에서 유의하게 개선됐다.
Depth 7→8의 OSD 개선은 BB72/BB144 p=.08에서만 유의했다. 이전 depth 5는
학습 schedule과 OSD 종류가 달라 아래 과거 결과와 직접 혼합하지 않는다.
재생성: `python scripts/analyze_bb_october24.py`.

Experiment directories keep their original `resdir_<id>` names and are grouped
by code family, noise model, and decoder:

```text
results/
├── toric/
│   ├── phenomenological/{cnn3d,convgru,pymatching}/
│   └── circuit/{convgru_mwpm,convgru_weighted_mwpm}/
├── bb/
│   ├── code_capacity/depolarizing/{orbit,tanner_cnn}/{bb72,bb144,replicates}/
│   └── circuit/{library_bp_osd,q_equals_p_idle0,no_osd,noise_balance,neural_relay}/
├── analysis/
├── plots/{bb/{code_capacity,circuit},toric/phenomenological}/
└── local_smoke/
```

## October 12, 2026: missing server jobs prepared for Git transfer

Jobs `58793764` (BB72 depth 1) and `58793765` (BB144 depth 1) were moved from
the archive to repository-root `resdir_58793764/` and `resdir_58793765/`, because
they were absent from the supplied server listing. `.gitignore` allows both
folders for transfer. All 100 files (40,297,607 bytes) retain identical SHA-256
hashes. [Relocation manifest](analysis/bb_server_missing_sources_2026_10_12.csv).
The three jobs in the earlier cleanup below remain archived.

## October 12, 2026: remaining root result folders archived

The three remaining root jobs are now inside the existing Tanner CNN archive:
`58793761` under `bb72/`, `58793763` under `bb144/`, and `58793766` under
`replicates/`. All 150 files (120,321,736 bytes) retain identical SHA-256 hashes.
Checkpoint discovery supports these paths; this cleanup initially left no root
`resdir_*` folders, before the two transfer jobs above were moved out.
[Relocation manifest](analysis/bb_root_cleanup_2026_10_12.csv).

## October 12, 2026: depth 4/5 continued training

[추가 학습 분석](analysis/bb_update_2026_10_12.md): jobs `59622483`, `59622488`,
`59622490`의 12개 모델이 모두 완료됐다. 11개는 총 800 epochs, 실패했던
BB144 depth 5 p=.04는 총 400 epochs까지 복구됐다. 같은 final shots에서 비교하면
400→800 연장은 raw 10/11, 기존 residual OSD-0 7/11에서 유의한 개선을 보였다
(24개 검정 Holm 보정). 같은 800-epoch 예산으로 비교 가능한 5개 지점 모두
depth 5가 depth 4보다 좋다. BB72 depth 5 p=.04는 이전 best 가중치를 유지했다.

- [최종 성능](plots/bb/code_capacity/tanner_cnn/overview.png)
  · [같은 샘플에서의 개선량](analysis/bb_update_2026_10_12_cnn_resume_pairs.csv)
  · [학습 기록](analysis/bb_update_2026_10_12_cnn_validation.csv)
- [원본 3개 job](bb/code_capacity/depolarizing/tanner_cnn/continued_training/):
  165개 파일을 보존했고 이동 전후 SHA-256과 크기를 확인했다.
- [검증 요약](analysis/bb_update_2026_10_12_audit.json)
  · [이동 기록](analysis/bb_update_2026_10_12_relocation.csv).

재생성: `python scripts/analyze_bb_october12.py`.
아래 표준 syndrome OSD-0 재평가와는 별도 결과다.

## October 12, 2026: frozen CNN with standard syndrome OSD-0

[재학습 없는 OSD 비교](analysis/bb_cnn_standard_osd0_2026_10_12.md): BB144 depth 5,
p=.06, 400-epoch 모델의 기존 65,536 final shots를 CPU에서 재평가했다.
Residual OSD-0 370 failures, syndrome OSD-0 366 failures로, 이 한 지점에서는
유의한 차이가 확인되지 않았다. 과거 GPU 저장값과의 작은 재현 차이는 보고서에
별도 기록했다. 전체 모델을 평가한 결과는 아니다.
실행법은 [post-hoc 평가 안내](../docs/bb_cnn_posthoc_osd.md)를 참고한다.

## October 10, 2026: 400-epoch CNN depth sweep and low-p circuit statistics

**400-epoch 분석은 [10월 10일 보고서](analysis/bb_update_2026_10_10.md)와
[depth 2–5 비교](plots/bb/code_capacity/tanner_cnn/overview.png)에서 확인한다.**

- CNN 36개 중 35개 완료. Depth 5는 비교 가능한 7개 지점 모두 raw LER 최저;
  p≥.04의 완료된 5개 지점은 CNN+OSD도 최저다. Depth 2는 100+300 LR restart,
  depth 3–5는 fresh 400-epoch cosine이므로 학습 schedule은 다르다.
- [학습 기록](analysis/bb_update_2026_10_10_cnn_validation.csv)과
  [최종 수치](analysis/bb_update_2026_10_10_cnn_final.csv). BB144 depth 5 p=.04는
  114 epochs 저장 후 validation OSD 확률 검증 오류로 실패해 final 결과가 없다.
- BB72 circuit p=.001/.002는 각각 100만 shots 완료:
  [그림](plots/bb/circuit/library_bp_osd/lowp_million.png),
  [집계](analysis/bb_update_2026_10_10_baseline_final.csv).
  Chunks는 다운로드에서 제외해 aggregate 검증만 했다.
- BB144 circuit 10개 실행은 OOM/time limit 로그와 저장된 처리 shot 0을 확인했다.
  LER이 0이라는 뜻이 아니며 성능 곡선에 넣지 않았다.

새 12개 `resdir_*`는 내부 파일과 job 이름을 보존한 채 아래로 옮겼다.

- CNN 9개: [`bb/code_capacity/depolarizing/tanner_cnn/400_epochs/`](bb/code_capacity/depolarizing/tanner_cnn/400_epochs/)
  아래 `bb72/`, `bb144/`, `replicates/`.
- Circuit BB144 2개: [`bb/circuit/library_bp_osd/bb144/`](bb/circuit/library_bp_osd/bb144/).
- Circuit 혼합-code 추가 통계 1개: [`bb/circuit/library_bp_osd/replicates/`](bb/circuit/library_bp_osd/replicates/).

644개 파일의 이동 전후 SHA-256과 크기가 일치한다.
[이동 기록](analysis/bb_update_2026_10_10_relocation.csv),
[SHA-256 manifest](analysis/bb_update_2026_10_10_manifest.csv)와
[검증 결과](analysis/bb_update_2026_10_10_audit.json)를 저장했다.
기존 9월 분석과 그림은 보존했다.

```bash
python scripts/analyze_bb_october10.py
```

## Figures: start here

**[CNN 통합 성능 그림](plots/bb/code_capacity/tanner_cnn/overview.png) 하나에서
각 설정의 최신 결과 44개를 비교한다.** 색은 depth다. 추가 학습 이전 단계는 표시하지 않는다.
학습·seed·개별 시기 그림은 정리했고 원본 데이터와 통계는 보존했다.
[표시된 최신 수치](analysis/bb_cnn_latest_final.csv). 재생성: `python scripts/plot_bb_cnn_all.py`.

수정한 circuit BP/OSD baseline의 새 BB72 결과는
[전용 성능 그림](plots/bb/circuit/library_bp_osd/overview.png)과
[9월 30일 분석](analysis/bb_library_bp_osd_2026_09_30.md)에 정리했다.

The [plot guide / 플롯 안내](plots/README.md) explains every figure, Block LER,
and regeneration commands. Figures are grouped by **code → noise → model**:

| Experiment | Figure folder | Main figure |
| --- | --- | --- |
| Current joint Tanner CNN, capacity | [bb/code_capacity/tanner_cnn/](plots/bb/code_capacity/tanner_cnn/) | [overview](plots/bb/code_capacity/tanner_cnn/overview.png) |
| Current library BP / BP+OSD, circuit | [bb/circuit/library_bp_osd/](plots/bb/circuit/library_bp_osd/) | [overview](plots/bb/circuit/library_bp_osd/overview.png) |
| Previous Neural BP4, capacity | [bb/code_capacity/neural_bp/](plots/bb/code_capacity/neural_bp/) | [overview](plots/bb/code_capacity/neural_bp/overview.png) |
| Relay BP / Neural Relay BP / Neural BP / Neural+OSD, circuit | [bb/circuit/neural_bp/](plots/bb/circuit/neural_bp/) | [combined overview](plots/bb/circuit/neural_bp/overview.png) |
| Toric ConvGRU / PyMatching | [toric/phenomenological/](plots/toric/phenomenological/) | [threshold](plots/toric/phenomenological/threshold.png) |

The Tanner CNN overview shows only the latest completed run per code/p/depth/seed
in four panels (BB72/BB144 × raw/residual OSD-0). Color identifies depth. Earlier
training stages are omitted; 44 configurations remain. Extra seeds and the
400-epoch recovery are marked separately. Training budgets differ across points.
Prior Neural BP4 appears in the raw panels only.
Circuit `neural_bp/overview.png` combines Relay BP, Neural Relay BP, Neural BP,
and Neural BP+OSD-0 in two panels (BB72 / BB144). These curves share the original
idle-free circuit and X/Z block-failure metric; original histories and DEM
fingerprints are checked before plotting. Both Relay methods have eight points
per code. Neural BP has four points per code; Neural+OSD has eight BB72 and four
BB144 points. Every point is a completed 4,096-shot final evaluation.
Missing raw points are not filled from OSD-selected checkpoints.

Relay allows up to 4 × 12 BP iterations and a 2-solution search, versus 12
iterations for Neural BP. Training and checkpoint selection differ, and
Neural+OSD retains its original posterior-seeded wrapper. This is a comparison
of the recorded methods, not a matched-compute or same-checkpoint ablation.
The corrected `library_bp_osd/overview.png` remains separate because its circuit
noise and observable task differ.

The old raw/OSD PNG and all three separate Relay PNGs were removed. Training,
validation, paired statistics, and ablation tables remain in `analysis/`, and
all original experiment files remain in `bb/circuit/`. There are no PDFs.

Known partial runs are retained in `cnn3d/resdir_55562044`,
`cnn3d/resdir_55860108`, and `convgru/resdir_1252167/exp_6`.

## September 30, 2026: corrected BB72 circuit BP/OSD baselines

Jobs `58939136` (p=.001–.004) and `58939138` (p=.005–.008) are archived under
[`bb/circuit/library_bp_osd/bb72/`](bb/circuit/library_bp_osd/bb72/).
All eight server experiments completed 100,000 shots, yielding 48 decoder/input
rows on 800,000 unique circuit shots. They use unscaled min-sum BP, at most
1000 iterations, direct library OSD-0/CS3, and six noisy memory cycles.

The [import manifest](analysis/bb_library_bp_osd_2026_09_30_manifest.csv) verifies
6,201 downloaded files (320,688,419 bytes) before and after relocation.
Final JSON/CSV aggregates are complete. The p=.008 local download has missing
chunks/provenance and an unreadable chunk; this is distinct from an incomplete
server experiment. Seven points have full independent correction rescoring;
the eighth has a documented partial audit. No missing file is fabricated.

At p=.004, X-only logical-Z error rates are BP **8.176%**, OSD-0 **6.198%**,
OSD-CS3 **5.342%**; joint-XZ OSD-CS3 is **9.344%**. Joint input helps the OSD
branches at p=.001/.002 but degrades them from p=.003 onward in this sweep.
Plain BP's logical-prediction metric and syndrome-inclusive block failure are
reported separately. Old Neural/Relay results use a different circuit and
observable task. Current BP is plotted only in `library_bp_osd/`; the separate
Relay/Neural overview contains the original Relay control, but no current
library baseline. Original paired statistics remain in the analysis CSVs.

- [Korean analysis](analysis/bb_library_bp_osd_2026_09_30.md)
- [48 final rows](analysis/bb_library_bp_osd_2026_09_30.csv)
- [Paired comparisons](analysis/bb_library_bp_osd_2026_09_30_paired.csv)
- [Audit](analysis/bb_library_bp_osd_2026_09_30_audit.json) and
  [missing/unreadable download inventory](analysis/bb_library_bp_osd_2026_09_30_missing.csv)
- [Single overview PNG](plots/bb/circuit/library_bp_osd/overview.png)

```bash
python scripts/summarize_bb_library_baselines.py
```

## September 27, 2026: complete Tanner CNN campaign and baseline replacement

The obsolete scale-0.625 plain-BP jobs `58793753`, `58793754`, `58793756`,
`58793757`, `58793759` and their three aggregate CSVs/PNG were deleted at the
user's request. The [deletion record](analysis/bb_plain_bp_removal_2026_09_27.json)
lists the exact removed directories. Previous Neural/Relay data is retained.
The new [library BP/OSD campaign](../docs/bb_baseline_campaign.md) uses the paper
circuit schedule, idle noise and logical-X memory task. Its BB72 results were
imported on September 30 above; BB144 and the extra low-p statistics are not yet archived here.

Four new jobs (`58793763`--`58793766`) join retained job `58793761` under
[`bb/code_capacity/depolarizing/tanner_cnn/`](bb/code_capacity/depolarizing/tanner_cnn/).
They cover both BB72 and BB144, depth 1/2 at p=.02/.04/.06/.08, and three
depth-2 seeds per code at p=.06. All 20 runs completed 100 epochs and 65,536
final shots. The [manifest](analysis/bb_import_2026_09_27_manifest.csv) verifies
all 250 original files (160,619,343 bytes). Saved raw/OSD corrections are
independently rescored. Eight depth pairs share byte-identical truth/syndrome
banks: 1,310,720 model-shot evaluations represent 786,432 unique final shots.

Depth 2 improves on depth 1 at every point, for both raw and OSD output, with
paired significance after Holm correction. At p=.06, primary-seed raw/OSD
LER is 33.841%/8.684% for BB72 and 58.130%/3.600% for BB144. The three-seed
OSD means are 8.546% and 3.497%, respectively. Raw failures are overwhelmingly
syndrome-invalid. CNN+OSD still loses to previous Neural BP4 at all shared
p=.04/.06/.08 points; the older neural model had 12 times the training samples.
BB144 p=.02 has only one OSD failure, so its rare-event estimate remains broad.
These are capacity results, not the pending circuit baseline results.

See the [Korean analysis](analysis/bb_update_2026_09_27.md),
[final results](analysis/bb_update_2026_09_27_cnn_final.csv),
[paired depth comparison](analysis/bb_update_2026_09_27_cnn_depth_pairs.csv),
[seed summary](analysis/bb_update_2026_09_27_cnn_seeds.csv), and
[audit](analysis/bb_update_2026_09_27_audit.json).

Three CNN PNGs are consolidated in
[`plots/bb/code_capacity/tanner_cnn/`](plots/bb/code_capacity/tanner_cnn/); no PDF is generated.
Reproduce the audit, six analysis CSVs and three PNGs with:

```bash
python scripts/summarize_bb_september27.py
```

## September 23, 2026: ordinary BP evaluation scripts

The September 23 version of `run_bb_0.slurm` through `run_bb_4.slurm` launched
a max-12/max-1000 BP sweep on 4,096 circuit shots without OSD. That campaign
was superseded on September 27 and its invalid baseline data was deleted as
recorded above. The current scripts launch the
[library BP/OSD campaign](../docs/bb_baseline_campaign.md).

## September 22, 2026: BP baseline stopping and iteration caps

New BB circuit evaluations report ordinary BP with caps 12 and 1000 by default,
on the same circuit samples. Each shot stops at its first syndrome-valid
correction. The short cap follows `--bp_iterations`; the additional cap is
`--bb_bp_reference_iterations`. Both references, including LER and mean actual
iterations, are saved in each evaluation's `bp_baselines` fields in
`history.json`. Relay runs retain their paired non-neural Relay comparison.
Non-Relay neural evaluation stops within its trained depth, while training
continues full unrolling. OSD, when enabled, processes only unconverged shots.

Archived results and plots have not been recalculated. New histories record
`bp_evaluation_policy=first_syndrome_valid_v1`; old checkpoint-selection histories
cannot be resumed under the changed policy. These baseline changes preserve
the configured min-sum scaling and noise model, so they do not establish
literature-matched LERs. See the [decoder documentation](../docs/bb_neural_relay_bp.md).

## September 17, 2026: non-relay BP clipping fix

`EquivariantNeuralBP2` now subtracts the recipient check message before clipping
the outgoing variable message. This corrects saturated messages and their
gradients in both the plain and neural non-relay paths. Regression tests cover
positive/negative saturation, degree-one variables, and extrinsic gradients.
The September 16 measurements remain results of the old implementation; the
audit script explicitly replays that old update when generating archived rows.
Re-evaluating an old non-relay checkpoint with the current code uses the fixed
update and can produce different outputs.

## September 16, 2026: circuit baseline implementation audit

The [implementation audit](analysis/bb_circuit_baseline_audit_2026_09_16.md)
distinguishes the archived circuit baselines from literature reproductions.
The archived OSD wrapper reseeds an additional BP pass with the supplied
posterior; it does not hand that posterior directly to OSD. The older non-relay
variable update also clips before removing the recipient check message. The
Relay implementation already removes the recipient before clipping.
Noise, logical observables, OSD order, and iteration budgets differ from the
papers. Existing runs and plots retain their original values and describe
comparisons within those settings, not a matched literature benchmark.

The [audit script](../scripts/audit_bb_circuit_baseline.py) saves a shared circuit
shot bank, decoder outcomes, numerical counterexamples, and direct `ldpc`
BP-OSD reference measurements. It leaves the training decoder unchanged.

## September 15, 2026: BB144 Neural Relay results

The newly imported `resdir_58164812` and `resdir_58164813` (jobs 2 and 3)
are archived with their original names under
[`bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb144/`](bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb144/).
All eight runs completed 100 epochs and a fresh 4,096-shot selected-best
evaluation, without OSD. The 102 original files, including checkpoints, were
verified by SHA-256 before and after moving them into the archive.

The [BB144 report](analysis/bb_neural_relay_bb144_2026_09_15.md) includes the
[final comparison CSV](analysis/bb_neural_relay_bb144_2026_09_15.csv),
[training records](analysis/bb_neural_relay_bb144_2026_09_15_train.csv),
[validation records](analysis/bb_neural_relay_bb144_2026_09_15_validation.csv), and
[original-file manifest](analysis/bb_neural_relay_bb144_2026_09_15_manifest.csv).
Only p=0.005 shows a significant improvement, from 98.54% to 93.14% LER.
p=0.001/0.004 degrade after Holm correction across the eight BB144 tests.
Across all 16 BB72+BB144 tests, the p=0.004 degradation remains significant,
while p=0.001 does not. All observed BB144 Neural failures are syndrome invalid;
seven of eight runs end with higher average training loss than they started.

The [combined CSV](analysis/bb_neural_relay_bb72_bb144_2026_09_15.csv) and
[comparison figure](plots/bb/circuit/neural_bp/overview.png) retain
both codes' final LER curves. Multiple-testing scopes and paired gain remain in the CSV. Incomplete historical
BB144 high-p OSD runs remain marked partial and have no final LER substituted.
The single Relay/Neural overview is saved as PNG only.

Reproduce numerical extraction, audits, file manifests, and figures for both codes:

```bash
python scripts/summarize_bb_neural_relay.py --code all
```

Use `--code bb144` for only the new results. Markdown interpretation is maintained
in the report rather than automatically overwritten by the script.

## September 13, 2026: first Neural Relay results

`resdir_58164810` and `resdir_58164811` contain the completed BB72 low/high
sweeps from jobs 0 and 1. Both are archived under
[`bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb72/`](bb/circuit/neural_relay/q_equals_p_idle0/orbit/bb72/).
All eight points have 100 epochs and 4,096 fresh selected-best shots.
On September 15, the 102 original files were moved from the repository root
and verified against the [SHA-256 manifest](analysis/bb_neural_relay_bb72_2026_09_13_manifest.csv).
Report and CSV source paths point to the archive. BB144 Relay results were
added separately on September 15, as described above.

The [analysis](analysis/bb_neural_relay_bb72_2026_09_13.md),
[final comparison CSV](analysis/bb_neural_relay_bb72_2026_09_13.csv), and
[overview figure](plots/bb/circuit/neural_bp/overview.png) distinguish
non-neural Relay from the historical single-pass BP baseline. Learning improves
LER at p=0.004/0.005/0.008 after exact paired tests with Holm correction.
The historical Neural+OSD pipeline still has lower LER; later training degrades
several Relay models. This is one training seed per point, not a threshold study.

Regenerate and audit the numerical extraction and figures with:

```bash
python scripts/summarize_bb_neural_relay.py --code bb72
```

## Recent ConvGRU resume sweep

`toric/phenomenological/convgru/resdir_1084621` and `resdir_1085072`
contain the completed learning-rate sweep for resumed pure-ConvGRU models.
The extracted comparison, including source checkpoints and PyMatching
baselines, is in `analysis/convgru_resume_lr_sweep.csv`; conclusions are in
`analysis/convgru_resume_lr_sweep.md`.

## Recent pure ConvGRU threshold-gap sweep

`toric/phenomenological/convgru/resdir_1116847` and `resdir_1117203`
contain eight completed L13/L15 runs filling the narrow p-grid from 0.011 to
0.019. Extracted settings and metrics are in
[`analysis/convgru_threshold_gap_2026_08.csv`](analysis/convgru_threshold_gap_2026_08.csv),
with interpretation in
[`analysis/convgru_threshold_gap_2026_08.md`](analysis/convgru_threshold_gap_2026_08.md).

Three runs stopped near a cross-entropy loss of `ln(4)` and are classified as
optimization plateaus. They remain in the curated threshold CSV and plot, but
are shown as `X` markers rather than connected threshold-curve points.

## BB Neural BP depolarizing sweep

The completed orbit-shared Neural BP sweeps are archived as
`bb/code_capacity/depolarizing/orbit/bb72/resdir_57181711` and
`bb/code_capacity/depolarizing/orbit/bb144/resdir_57181713`. Together they
cover `p=0.04,0.06,0.08,0.10`; every point has 300 epochs and a fresh
131,072-shot selected-best evaluation against vanilla BP4 on the same samples.

The extracted data and interpretation are in
[`analysis/bb_neural_bp_depolarizing_orbit.csv`](analysis/bb_neural_bp_depolarizing_orbit.csv)
and
[`analysis/bb_neural_bp_depolarizing_orbit.md`](analysis/bb_neural_bp_depolarizing_orbit.md).
Its early BP-only figure has been retired because the newer same-bank decoder
comparison below contains the Neural/vanilla curves together with the stronger
OSD and LSD baselines.

## August 2026 BB campaign

The ten `run_bb_0`--`run_bb_9` jobs are archived by purpose under:

```text
bb/code_capacity/
├── depolarizing/
│   ├── orbit/replicates/
│   ├── ablations/{sharing,iterations,mechanism,loss_auxiliary,loss_core}/
│   └── classical/css_separated/{bb72,bb144}/
└── independent_xz/marginal_matched/orbit/
```

Completion must be determined from each output's `training_log.txt` and
`history.json`, not only the top-level srun exit code. Of 32 Neural BP runs, 28
have a fresh 131,072-shot `[Selected Best]` evaluation and four are genuinely
partial. All eight classical `(code,p)` points completed.

The complete extraction and interpretation are in
[`analysis/bb_campaign_2026_08.md`](analysis/bb_campaign_2026_08.md). Its
machine-readable inputs are
[`analysis/bb_campaign_2026_08_neural.csv`](analysis/bb_campaign_2026_08_neural.csv)
and
[`analysis/bb_campaign_2026_08_classical.csv`](analysis/bb_campaign_2026_08_classical.csv).
The selected-best Neural/BP4 checkpoints are also evaluated on the exact saved
classical error banks in
[`analysis/bb_neural_vs_classical_paired.csv`](analysis/bb_neural_vs_classical_paired.csv).
Rebuild the summaries and plots with:

```bash
python scripts/summarize_bb_campaign.py
```

The paired same-bank CSV itself is reproduced with:

```bash
python scripts/bb_neural_vs_classical_paired.py
```

The main figures are
[`plots/bb/code_capacity/neural_bp/ablations.png`](plots/bb/code_capacity/neural_bp/ablations.png)
and
[`plots/bb/code_capacity/neural_bp/overview.png`](plots/bb/code_capacity/neural_bp/overview.png).

## August–September 2026 BB circuit-level campaign

The corrected circuit-schema-v2 campaign is archived by purpose:

```text
bb/circuit/
├── q_equals_p_idle0/
│   ├── orbit/{bb72,bb144,replicates}/
│   └── ablations/{sharing,iterations,mechanism,loss_auxiliary}/
├── no_osd/q_equals_p_idle0/orbit/{bb72,bb144}/
└── noise_balance/orbit/
```

The archive contains 40 histories: 35 completed all 100 epochs plus a fresh
4,096-shot selected-best OSD-0 evaluation, while five BB144 runs reached the
24-hour Slurm limit and are retained as resumable partial runs. The primary
reference sweep contains eight complete BB72 points (`p=0.001` through
`0.010`) and four complete BB144 points (`p=0.001` through `0.004`). The four
higher-p BB144 reference points are partial.

The new campaign also contains three training seeds per code at `p=0.004`,
global-versus-orbit sharing, T=6/12/24, learned-residual and relaxation
component ablations, auxiliary-loss ablations, and `idle=p`/`q=2p` noise
controls. The headline comparison is Neural-BP posterior + OSD-0 versus
vanilla-BP posterior + the same OSD-0 post-processor; plain BP remains a
convergence diagnostic.

The complete extraction and interpretation are in
[`analysis/bb_circuit_campaign_2026_08.csv`](analysis/bb_circuit_campaign_2026_08.csv)
and
[`analysis/bb_circuit_campaign_2026_08.md`](analysis/bb_circuit_campaign_2026_08.md).
Interrupted-run provenance and resume information are in
[`analysis/bb_circuit_campaign_2026_08_partial.csv`](analysis/bb_circuit_campaign_2026_08_partial.csv).
Rebuild the summaries and plots with:

```bash
python scripts/summarize_bb_circuit_campaign.py
```

The shared Relay/Neural figure is
[`plots/bb/circuit/neural_bp/overview.png`](plots/bb/circuit/neural_bp/overview.png).
Ablation results remain in the analysis report and CSV.
This is still not a threshold estimate: BB144 high-p completion, more training
seeds, a larger fixed test bank, and stronger OSD-CS/LSD comparisons remain
necessary.

## September 2026 BB circuit-level no-OSD diagnostic

The two raw-decoder jobs are archived as
`bb/circuit/no_osd/q_equals_p_idle0/orbit/bb72/resdir_57890809` and
`bb/circuit/no_osd/q_equals_p_idle0/orbit/bb144/resdir_57890812`. All eight
points (`p=0.001` through `0.004` for each code) completed 100 epochs and a
fresh 4,096-shot selected-best evaluation. They use raw Neural BP2 versus raw
vanilla BP2 on paired Stim samples; no OSD evaluation or repair is present.

The extracted data and interpretation are in
[`analysis/bb_circuit_no_osd_2026_09.csv`](analysis/bb_circuit_no_osd_2026_09.csv)
and
[`analysis/bb_circuit_no_osd_2026_09.md`](analysis/bb_circuit_no_osd_2026_09.md).
The empty partial-run inventory is
[`analysis/bb_circuit_no_osd_2026_09_partial.csv`](analysis/bb_circuit_no_osd_2026_09_partial.csv).
Rebuild the raw summary tables with:

```bash
python scripts/summarize_bb_circuit_no_osd.py
```

The separate raw-only figure has been retired because the combined figure below
contains the same raw failure curve; paired gains remain in the report and CSV. These points are a
convergence diagnostic, not a threshold curve: raw success is almost identical
to syndrome convergence, and BB144 collapses for `p>=0.002` without a global
repair stage.

The common `p=0.001`--`0.004` raw and OSD-assisted results are overlaid in
[`plots/bb/circuit/neural_bp/overview.png`](plots/bb/circuit/neural_bp/overview.png),
with the merged values in
[`analysis/bb_circuit_raw_vs_osd_2026_09.csv`](analysis/bb_circuit_raw_vs_osd_2026_09.csv).
Recreate both with:

```bash
python scripts/plot_bb_circuit_raw_vs_osd.py
```

This overlay is descriptive rather than a clean OSD ablation. The raw campaign
selects checkpoints by raw paired gain, while the older OSD campaign selects
them by OSD paired gain. A causal before/after comparison requires applying OSD
to the same frozen checkpoint on the same saved test bank.
