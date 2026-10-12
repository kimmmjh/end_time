"""Direct order-zero ordered-statistics repair of binary hard decisions.

No BP is constructed or executed. Gaussian elimination chooses independent
columns in ascending |LLR| order and changes only those least-reliable basis
bits to solve the residual syndrome. Nonbasis bits retain the input hard
decision. This hard-decision-centred OSD-0 convention is explicit; it need not
match ldpc's zero-free-variable syndrome-decoding convention bit for bit.
"""

from __future__ import annotations

import numpy as np


class DirectOSD0:
    def __init__(self, check_matrix):
        matrix = check_matrix.toarray() if hasattr(check_matrix, "toarray") else np.asarray(check_matrix)
        if matrix.ndim != 2 or not np.isin(matrix, [0, 1]).all():
            raise ValueError("check_matrix must be a binary matrix.")
        self.check = np.asarray(matrix, dtype=np.uint8).copy()

    def decode(self, syndrome, probabilities):
        syndrome = np.asarray(syndrome)
        probabilities = np.asarray(probabilities, dtype=np.float64)
        if syndrome.shape != (self.check.shape[0],) or not np.isin(syndrome, [0, 1]).all():
            raise ValueError("syndrome must be a binary vector matching the check rows.")
        if (probabilities.shape != (self.check.shape[1],) or not np.isfinite(probabilities).all()
                or ((probabilities < 0) | (probabilities > 1)).any()):
            raise ValueError("probabilities must be a finite [0,1] vector matching the columns.")
        hard = (probabilities > 0.5).astype(np.uint8)
        residual = syndrome.astype(np.uint8) ^ ((self.check @ hard) % 2)
        if not residual.any():
            return hard
        clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
        reliability = np.abs(np.log1p(-clipped) - np.log(clipped))
        order = np.argsort(reliability, kind="stable")
        matrix = self.check[:, order].copy()
        rhs = residual.copy()
        pivots = []
        rank = 0
        for column in range(matrix.shape[1]):
            candidates = np.flatnonzero(matrix[rank:, column])
            if not candidates.size:
                continue
            pivot = rank + int(candidates[0])
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
            rhs[[rank, pivot]] = rhs[[pivot, rank]]
            eliminate = np.flatnonzero(matrix[:, column])
            eliminate = eliminate[eliminate != rank]
            matrix[eliminate] ^= matrix[rank]
            rhs[eliminate] ^= rhs[rank]
            pivots.append(column)
            rank += 1
            if rank == matrix.shape[0]:
                break
        if rhs[rank:].any():
            raise ValueError("syndrome is not in the image of the check matrix.")
        hard[order[np.asarray(pivots, dtype=np.int64)]] ^= rhs[:rank]
        if not np.array_equal((self.check @ hard) % 2, syndrome):
            raise RuntimeError("OSD-0 failed the syndrome check.")
        return hard

    def decode_batch(self, syndrome, probabilities):
        syndrome, probabilities = np.asarray(syndrome), np.asarray(probabilities)
        if syndrome.ndim != 2 or probabilities.shape != (syndrome.shape[0], self.check.shape[1]):
            raise ValueError("Expected syndrome [batch,m] and probabilities [batch,n].")
        return np.stack([self.decode(s, p) for s, p in zip(syndrome, probabilities)])


__all__ = ["DirectOSD0"]
