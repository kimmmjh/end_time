"""Experimental bit-packed OSD-0 on a Torch CPU/CUDA device.

This preserves DirectOSD0's stable ascending-|LLR| basis and hard decisions.
NumPy sorting and sparse syndrome products remain on the host. GF(2) row
elimination uses packed uint8 tensors on the requested device. It is a benchmark
candidate, not the production backend, and is not differentiable.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def packed_rank(packed: np.ndarray, columns: int) -> int:
    """Exact GF(2) rank, computed once at setup using packed CPU row operations."""
    matrix = packed.copy()
    rows = len(matrix)
    rank = 0
    for column in range(columns):
        if rank == rows:
            break
        byte, mask = column // 8, 1 << (column % 8)
        candidates = np.flatnonzero(matrix[rank:, byte] & mask)
        if not candidates.size:
            continue
        pivot = rank + int(candidates[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        # Forward elimination suffices for rank; decoding below uses full RREF.
        targets = np.flatnonzero(matrix[rank+1:, byte] & mask) + rank + 1
        matrix[targets] ^= matrix[rank]
        rank += 1
    return rank


class PackedDirectOSD0:
    """Same correction rule as DirectOSD0; GPU elimination with explicit device.

    Samples are currently decoded sequentially. CUDA speedup is not promised:
    pivot selection synchronizes the host, and these eager Torch operations
    launch many kernels. The benchmark includes these costs and transfers.
    """

    def __init__(self, check_matrix, *, device="cuda"):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Packed OSD supports CPU and CUDA devices only")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA OSD requested but CUDA is unavailable; no CPU fallback")
        matrix = check_matrix.toarray() if sp.issparse(check_matrix) else np.asarray(check_matrix)
        if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
            raise ValueError("check_matrix must be a binary matrix")
        matrix = np.asarray(matrix, dtype=np.uint8)
        self.check = sp.csr_matrix(matrix)
        self.rows, self.columns = matrix.shape
        packed = np.packbits(matrix, axis=1, bitorder="little")
        self.rank = packed_rank(packed, self.columns)
        self.packed = torch.as_tensor(packed.copy(), device=self.device)
        self.row_ids = torch.arange(self.rows, device=self.device)

    @torch.no_grad()
    def decode(self, syndrome, probabilities):
        syndrome = np.asarray(syndrome)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if syndrome.shape != (self.rows,) or not np.isin(syndrome, [0, 1]).all():
            raise ValueError("syndrome must be a binary vector matching the check rows")
        if probabilities.shape != (self.columns,):
            raise ValueError("Probability shape must match the check columns")
        if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Probabilities must be finite and in [0,1]")
        hard = (probabilities > .5).astype(np.uint8)
        residual = syndrome.astype(np.uint8) ^ (self.check @ hard % 2)
        if not residual.any():
            return hard
        clipped = np.clip(probabilities, 1e-12, 1-1e-12)
        reliability = np.abs(np.log1p(-clipped) - np.log(clipped))
        order = np.argsort(reliability, kind="stable")
        matrix = self.packed.clone()
        rhs = torch.as_tensor(residual.copy(), device=self.device)
        pivots = []
        rank = 0
        # Keep columns in their original packed positions; visiting them in
        # sorted order is algebraically identical to permuting H first.
        for column in order:
            if rank == self.rank:
                break
            column = int(column)
            byte, mask = column // 8, 1 << (column % 8)
            candidates = (self.row_ids >= rank) & ((matrix[:, byte] & mask) != 0)
            pivot = int(torch.where(candidates, self.row_ids, self.rows).amin().item())
            if pivot == self.rows:
                continue
            if pivot != rank:
                saved_row, saved_rhs = matrix[rank].clone(), rhs[rank].clone()
                matrix[rank].copy_(matrix[pivot])
                rhs[rank].copy_(rhs[pivot])
                matrix[pivot].copy_(saved_row)
                rhs[pivot].copy_(saved_rhs)
            targets = ((matrix[:, byte] & mask) != 0).to(torch.uint8)
            targets[rank] = 0
            # All bits in each byte undergo the same GF(2) row operation.
            matrix.bitwise_xor_(targets[:, None] * matrix[rank:rank+1])
            rhs.bitwise_xor_(targets * rhs[rank])
            pivots.append(column)
            rank += 1
        if bool(rhs[rank:].any()):
            raise ValueError("syndrome is not in the image of the check matrix")
        hard[np.asarray(pivots, dtype=np.int64)] ^= rhs[:rank].cpu().numpy()
        if not np.array_equal(self.check @ hard % 2, syndrome):
            raise RuntimeError("Packed OSD-0 failed the syndrome check")
        return hard

    def decode_batch(self, syndrome, probabilities):
        syndrome, probabilities = np.asarray(syndrome), np.asarray(probabilities)
        if (syndrome.ndim != 2 or syndrome.shape[1] != self.rows or
                probabilities.shape != (len(syndrome), self.columns)):
            raise ValueError("Expected syndrome [batch,m] and probabilities [batch,n]")
        if len(syndrome) == 0:
            return np.empty((0, self.columns), dtype=np.uint8)
        return np.stack([self.decode(s, p) for s,p in zip(syndrome, probabilities)])
