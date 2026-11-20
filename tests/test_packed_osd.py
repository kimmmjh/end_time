import numpy as np
import pytest
import scipy.sparse as sp
import torch

from src._direct_osd import DirectOSD0
from src._packed_osd import PackedDirectOSD0, packed_rank
from src.bb_code import gf2_rank


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA hardware required"))])
def test_exact_corrections_random_rank_deficient_and_tied_inputs(device):
    rng = np.random.default_rng(20261119)
    for m,n in ((3,4), (8,17), (11,25), (5,3), (0,4), (3,0)):
        for _ in range(5):
            check = rng.integers(0, 2, size=(m,n), dtype=np.uint8)
            if m >= 3:
                check[-1] = check[0] ^ check[1]
            packed = np.packbits(check, axis=1, bitorder="little")
            assert packed_rank(packed, n) == gf2_rank(check)
            cpu, candidate = DirectOSD0(check), PackedDirectOSD0(sp.csr_matrix(check), device=device)
            faults = rng.integers(0, 2, size=(6,n), dtype=np.uint8)
            syndromes = (check @ faults.T).T % 2
            q = rng.choice([0., 1., .5, .1, .9, .500001, .499999], size=(6,n))
            np.testing.assert_array_equal(candidate.decode_batch(syndromes, q), cpu.decode_batch(syndromes, q))
            np.testing.assert_array_equal(candidate.decode_batch(syndromes, q), cpu.decode_batch(syndromes, q))


def test_errors_and_zero_residual():
    check = np.array([[1,0,1], [1,0,1], [0,0,0]], dtype=np.uint8)
    decoder = PackedDirectOSD0(check, device="cpu")
    with pytest.raises(ValueError, match="image"):
        decoder.decode([1,0,0], [.5]*3)
    with pytest.raises(ValueError, match="finite"):
        decoder.decode([0,0,0], [np.nan,0,0])
    with pytest.raises(ValueError, match="shape"):
        decoder.decode([0,0,0], [0,0])
    # Hard decisions already satisfy H; detector-free hard bits must survive.
    np.testing.assert_array_equal(decoder.decode([1,1,0], [.9,.9,.1]), [1,1,0])
    assert decoder.decode_batch(np.zeros((0,3)), np.zeros((0,3))).shape == (0,3)


def test_cuda_request_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        PackedDirectOSD0(np.eye(2), device="cuda")
