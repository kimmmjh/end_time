# BB DEM 후속 실험 — 2026-11-02

기존 hierarchical 결과는 **추가 학습 없이 기존 DEM CNN의 raw validation-selected best weights를 사용한 평가**다. 같은 forward/loss로 학습된 CNN 뒤에 confident local correction과 residual OSD completion을 붙였다. Hierarchical 전용 loss나 학습된 별도 residual network는 없다.

## 실행

저장소의 새 파일과 수정된 `scripts/resume_bb_dem_cnn.py`를 서버 `~/end_time`에 동기화한 뒤 실행한다. 기존 `resdir_*`는 서버 저장소 루트에 두면 된다.

```bash
cd ~/end_time
# 가중치 없이도 source/작업 분류를 확인할 수 있다.
BB_DRY_RUN=1 bash run_bb_dem_continue.slurm
BB_DRY_RUN=1 bash run_bb_dem_hierarchical_plateau.slurm

# 서로 독립적인 두 실험. 제출은 사용자가 실행한다.
sbatch run_bb_dem_continue.slurm
sbatch run_bb_dem_hierarchical_plateau.slurm
```

두 launcher 모두 기존 Perlmutter 설정(account `m5328_g`, regular, GPU node, 24h, 4 workers/node, GPU 1개·CPU 16개/worker, step memory 32G, `python/3.10` 및 `$PSCRATCH/envs/nde`)을 사용한다. `ds08152@gmail.com`으로 END/FAIL/ARRAY_TASKS 메일을 켰다. 각 array는 0–5이며 마지막 task의 빈 GPU slot은 실행하지 않는다. 동시 node 수 제한은 `sbatch --array=0-5%2 ...`로 조절할 수 있다. 24시간 이내 완료는 보장하지 않는다.

## 분류와 근거

| 코드 | Depth | 추가 학습할 p (%) | 바로 hierarchical 평가할 p (%) |
|---|---|---|---|
| BB72 | 4, 6, 8 | 0.1, 0.2, 0.4 | 0.6 |
| BB72 | 12 | 0.1, 0.2, 0.4, 0.6 | 없음 |
| BB144 | 4 | 0.1 | 0.2, 0.4, 0.6 |
| BB144 | 6, 8 | 0.1, 0.2 | 0.4, 0.6 |
| BB144 | 12 | 0.1, 0.2, 0.4 | 0.6 |

21개 모델은 추가 학습, 나머지 11개는 best/latest 두 가중치를 평가한다. 이는 **제한된 추가 학습 예산의 우선순위**이며, 나머지 모델이 더 학습해도 절대 개선되지 않는다는 판정이 아니다. Epoch 601–650과 751–800의 평균 raw validation failure 및 training logical loss를 manifest에 기록했다.

- BB72 낮은 p는 validation이 계속 개선된다. p=0.4%는 D4/D6/D8의 개선은 작지만 logical loss도 감소하므로 200 epoch 연장을 시험한다.
- BB72 D12 p=0.6%는 raw failure가 약 99.85→99.75%, logical loss가 0.6647→0.6607로 감소한다. 아직 실패율은 높지만 같은 p의 얕은 모델보다 연장을 시험할 근거가 있다.
- BB144 D4 p=0.2%는 raw failure 96.16→95.86%, loss 0.3316→0.3245로 개선이 약해 우선 평가한다. D6/D8/D12는 같은 p에서 더 뚜렷한 개선이 있어 연장한다.
- BB144 D12 p=0.4%는 raw failure는 거의 100%지만 loss가 0.6403→0.6235로 줄어 연장한다. D8은 0.6754→0.6711로 개선이 작고 best가 epoch 5에 남아 있으므로 latest 평가를 먼저 한다.
- BB144 p=0.6%는 모든 depth에서 raw failure 100%, logical loss 약 0.696으로 정체돼 먼저 평가한다.

[기존 분석](../results/analysis/bb_dem_2026_11_02.md) · Validation 곡선 (historical plot removed; numerical results retained) · Loss 곡선 (historical plot removed; numerical results retained)

## 추가 학습 동작

`run_bb_dem_continue.slurm`은 **latest `model.pt`의 epoch 800에서 1,000까지 200 epoch를 추가**한다. Best checkpoint로 되돌아가서 학습하지 않는다. 모델 가중치, AdamW state, 과거 history와 best를 보존하며 architecture, batch budget, loss weights, raw validation 기반 best 선택은 그대로다.

800 epoch cosine 종료 시 저장된 LR이 0일 수 있으므로 **LR=3e-5 (최초 3e-4의 1/10)에서 새 cosine 스케줄**을 시작한다. 이는 효과가 검증된 최적 LR이 아니라 제한적인 continuation 실험의 시작값이다. Sampler도 독립 phase로 시작하므로 중단 없는 1,000 epoch 학습과 동일하지 않다.

```bash
# 필요할 때 기본 예산/학습률 변경
sbatch --export=ALL,BB_DEM_TARGET_EPOCHS=1000,BB_DEM_RESTART_LR=0.00003 run_bb_dem_continue.slurm
```

추가 학습 launcher는 학습 및 기존 raw final 평가까지 실행하며, hierarchical 평가는 자동으로 이어 붙이지 않는다. 학습과 OSD 평가가 한 번의 24h 제한 안에서 경쟁하지 않도록 분리했다. 새 `model.pt`/`best_model.pt`는 후속 hierarchical 평가에 그대로 사용할 수 있다. Best 선택 기준은 여전히 raw accuracy이므로 추가 학습 후에도 best가 과거 epoch에 남을 수 있다.

## 바로 평가하는 동작

`run_bb_dem_hierarchical_plateau.slurm`은 11개 모델의 **best와 latest를 각각** 새 8,192 circuit shots로 평가한다(총 22 workers, 180,224 shots). 기본 confidence=0.9, batch=4, seed=20261101. 같은 모델의 두 가중치는 같은 noise/config/seed/batch로 shot stream을 생성한다. 샷 파일로 실제 일치를 확인할 수 있다. 기존 best 결과와 겹치는 모델도 같은 캠페인의 best/latest 비교를 남기기 위해 다시 평가한다.

각 worker는 raw/local/hierarchical을 동일한 샷에서 채점하고 metadata, results, batch별 `.npz`를 저장한다. 학습·optimizer 업데이트는 없다. 기존 24개 모델용 `run_bb_dem_hierarchical_eval.slurm`과 manifest는 재현성을 위해 변경하지 않았다. 새 manifest는 BB144 D8의 `resdir_60428756`, D12의 `resdir_60636624` 완료 run을 사용한다.

```bash
# 첫 4개 평가 worker에 대한 작은 pilot (학습은 하지 않음)
sbatch --array=0 --export=ALL,BB_HIER_SHOTS=128 run_bb_dem_hierarchical_plateau.slurm
```

## 출력과 중단 시 재개

출력은 학습·평가 모두 저장소 루트의 `resdir_<JOBID>`에 새로 생성한다. 앞으로 결과 폴더 이름에 작업 설명 접미사를 붙이지 않는다. 작업 종류는 `job_metadata.txt`의 campaign/mode에 기록한다. Source weights는 수정하지 않는다. 각 job에 source manifest, source snapshot, 실행 command, log, exit code를 저장한다.

- 학습: `worker_N/resume_plan.json`, `worker_N/outputs/<date>/<run>/{model.pt,best_model.pt,history.json,...}`
- 평가: `worker_N/{metadata.json,results.json,shots/batch_*.npz}`

기존 checkpoint가 없는 로컬 dry run은 `metadata_preview_only`이며 실제 optimizer 검증/학습은 하지 않는다. 서버 실행은 checkpoint 존재, manifest config, epoch와 history를 확인하고 학습 source Slurm job이 종료됐는지 확인한다.

시간 제한 후 원래 launcher를 그대로 재제출하면 **원래 epoch 800 source에서 다시 시작**하므로 그렇게 재개하지 않는다. 종료된 새 job의 `worker_N`을 `scripts/resume_bb_dem_cnn.py --source-job ... --code bb72|bb144 --depth D --index I --target-epochs 1000`에 지정해 남은 epoch만 재개한다. `I`는 p=0.001/0.002/0.004/0.006에 대응하는 0/1/2/3이다. Worker 디렉터리에 source job metadata도 복사한다. 중도 재개 때 `--learning-rate`를 생략하면 저장된 optimizer LR부터 시작한다. 실행은 GPU Slurm allocation 안에서 한다. 평가가 중단되면 새 출력 디렉터리에서 다시 평가한다.

## Array task / worker 목록

각 task는 연속 4개 worker를 실행한다. 마지막 train task는 1개, evaluate task는 2개 worker만 있다.

### train

| Array task | Worker | 코드 | Depth | p (%) | Weights |
|---:|---:|---|---:|---:|---|
| 0 | 0 | bb72 | 4 | 0.1 | latest |
| 0 | 1 | bb72 | 4 | 0.2 | latest |
| 0 | 2 | bb72 | 4 | 0.4 | latest |
| 0 | 3 | bb144 | 4 | 0.1 | latest |
| 1 | 4 | bb72 | 6 | 0.1 | latest |
| 1 | 5 | bb72 | 6 | 0.2 | latest |
| 1 | 6 | bb72 | 6 | 0.4 | latest |
| 1 | 7 | bb144 | 6 | 0.1 | latest |
| 2 | 8 | bb144 | 6 | 0.2 | latest |
| 2 | 9 | bb72 | 8 | 0.1 | latest |
| 2 | 10 | bb72 | 8 | 0.2 | latest |
| 2 | 11 | bb72 | 8 | 0.4 | latest |
| 3 | 12 | bb144 | 8 | 0.1 | latest |
| 3 | 13 | bb144 | 8 | 0.2 | latest |
| 3 | 14 | bb72 | 12 | 0.1 | latest |
| 3 | 15 | bb72 | 12 | 0.2 | latest |
| 4 | 16 | bb72 | 12 | 0.4 | latest |
| 4 | 17 | bb72 | 12 | 0.6 | latest |
| 4 | 18 | bb144 | 12 | 0.1 | latest |
| 4 | 19 | bb144 | 12 | 0.2 | latest |
| 5 | 20 | bb144 | 12 | 0.4 | latest |

### evaluate

| Array task | Worker | 코드 | Depth | p (%) | Weights |
|---:|---:|---|---:|---:|---|
| 0 | 0 | bb72 | 4 | 0.6 | best |
| 0 | 1 | bb72 | 4 | 0.6 | latest |
| 0 | 2 | bb144 | 4 | 0.2 | best |
| 0 | 3 | bb144 | 4 | 0.2 | latest |
| 1 | 4 | bb144 | 4 | 0.4 | best |
| 1 | 5 | bb144 | 4 | 0.4 | latest |
| 1 | 6 | bb144 | 4 | 0.6 | best |
| 1 | 7 | bb144 | 4 | 0.6 | latest |
| 2 | 8 | bb72 | 6 | 0.6 | best |
| 2 | 9 | bb72 | 6 | 0.6 | latest |
| 2 | 10 | bb144 | 6 | 0.4 | best |
| 2 | 11 | bb144 | 6 | 0.4 | latest |
| 3 | 12 | bb144 | 6 | 0.6 | best |
| 3 | 13 | bb144 | 6 | 0.6 | latest |
| 3 | 14 | bb72 | 8 | 0.6 | best |
| 3 | 15 | bb72 | 8 | 0.6 | latest |
| 4 | 16 | bb144 | 8 | 0.4 | best |
| 4 | 17 | bb144 | 8 | 0.4 | latest |
| 4 | 18 | bb144 | 8 | 0.6 | best |
| 4 | 19 | bb144 | 8 | 0.6 | latest |
| 5 | 20 | bb144 | 12 | 0.6 | best |
| 5 | 21 | bb144 | 12 | 0.6 | latest |

