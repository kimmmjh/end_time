from dataclasses import asdict
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from models._bb_dem_cnn import BBDemCNN
from scripts import evaluate_bb_dem_threshold05 as runner
from scripts.run_bb_dem_sample_scaling import worker_lock
from src._bb_circuit_experiment import _graph_fingerprint
from src.bb_circuit_data import BBCircuitGenerator
from src.bb_stim_utils import CIRCUIT_SCHEMA_VERSION


@pytest.mark.parametrize('worker,code,name',[(0,'bb72','bb72_d8_p0.002'),(1,'bb144','bb144_d6_p0.002')])
def test_snapshot_and_requested_weights(tmp_path, monkeypatch, worker, code, name):
    gen = BBCircuitGenerator(code,rounds=1,gate_error_rate=.003,
                             circuit_noise_model='standard',batch_size=2,seed=37)
    model = BBDemCNN(gen.graph,width=4,depth=1)
    config = dict(architecture='bb_dem_cnn',noise_model='circuit',code=code,rounds=1,
                  width=4,depth=1,circuit_schema_version=CIRCUIT_SCHEMA_VERSION,
                  noise_profile=asdict(gen.noise_profile),graph_fingerprint=_graph_fingerprint(gen.graph))
    manifest = tmp_path/'manifest.json'
    entries = [dict(name=n,expected_config=config) for n in ('bb72_d8_p0.002','bb144_d6_p0.002')]
    manifest.write_text(json.dumps(dict(experiments=entries)))
    monkeypatch.setattr(runner,'MANIFEST',manifest)
    source = tmp_path/'campaign'/name/'continuous/training/model.pt'
    source.parent.mkdir(parents=True)
    torch.save(dict(format='bb_dem_cnn_v1',config=config,epoch=9,best_epoch=2,
                    model_state_dict=model.state_dict(),best_model_state_dict=model.state_dict()),source)
    original = source.read_bytes()
    output = tmp_path/'evaluation'
    result = runner.main(['--campaign-root',str(tmp_path/'campaign'),'--output',str(output),
                          '--worker',str(worker),'--device','cpu','--shots','3','--batch-size','2'])
    assert result['status']=='complete' and result['sources']['latest']['selected_epoch']==10
    assert result['sources']['latest']['weights']=='latest'
    assert result['confidence']==.5 and result['comparison']=='gt'
    assert original == source.read_bytes() == (output/'checkpoint.pt').read_bytes()
    if worker == 1:
        assert set(result['evaluations']) == {'latest'}
        assert result['best_vs_latest'] is None and not (output/'best').exists()
        return
    assert set(result['evaluations']) == {'best','latest'}
    assert result['sources']['best']['selected_epoch']==3
    assert result['sources']['best']['weights']=='best'
    assert result['best_vs_latest']['shots']==3
    metadata = [json.loads((output/choice/'metadata.json').read_text()) for choice in ('best','latest')]
    assert [m['confidence_comparison'] for m in metadata] == ['gt','gt']
    assert metadata[0]['source']['sha256'] == metadata[1]['source']['sha256']
    # A changed sample must be rejected, even if aggregate scores happened to match.
    batch = next((output/'latest/shots').glob('*.npz'))
    with np.load(batch) as saved:
        data = dict(saved)
    data['detectors'] = data['detectors'].copy()
    data['detectors'][0,0] ^= 1
    np.savez_compressed(batch,**data)
    with pytest.raises(RuntimeError,match='changed detectors'):
        runner.compare_shots(output/'best',output/'latest',3)


def test_active_training_lock_is_not_removed(tmp_path):
    experiment=tmp_path/'bb72_d8_p0.002'
    experiment.mkdir()
    with worker_lock(experiment):
        with pytest.raises(RuntimeError,match='Worker lock already exists'):
            runner.main(['--campaign-root',str(tmp_path),'--output',str(tmp_path/'out'),'--device','cpu'])
        assert (experiment/'worker.lock.d/owner.json').exists()


def test_launcher_preview(tmp_path):
    script=runner.REPO/'run_bb_dem_threshold05_eval.slurm'
    subprocess.run(['bash','-n',str(script)],check=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('BB_')}
    env.update(BB_DRY_RUN='1',BB_PYTHON=sys.executable,BB_SCALE_RUN_DIR=str(tmp_path))
    output=subprocess.run(['bash',str(script)],env=env,check=True,capture_output=True,text=True).stdout
    plans=[]
    while output.strip():
        plan,end=json.JSONDecoder().raw_decode(output.lstrip())
        plans.append(plan)
        output=output.lstrip()[end:]
    assert [p['weights'] for p in plans]==[['latest','best'],['latest']]
    assert [p['experiment'] for p in plans]==['bb72_d8_p0.002','bb144_d6_p0.002']
    assert all(p['confidence']==.5 and p['shots']==8192 and not p['training']
               and not p['automatic_resubmission'] for p in plans)
    assert not list(tmp_path.iterdir())
