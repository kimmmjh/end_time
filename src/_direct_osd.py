"""Order-zero ordered-statistics decoding with two explicit conventions.

No BP is constructed or executed. Gaussian elimination chooses independent
columns in ascending |LLR| order and changes only those least-reliable basis
bits to solve the residual syndrome. Nonbasis bits retain the input hard
decision. This hard-decision-centred OSD-0 convention is explicit; it need not
match ldpc's zero-free-variable syndrome-decoding convention bit for bit.
SyndromeOSD0 implements the published syndrome variant with zero free bits.
"""

from __future__ import annotations

import numpy as np


class DirectOSD0:
    _hard_centered = True

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
        if probabilities.shape != (self.check.shape[1],):
            raise ValueError(f"Probability shape {probabilities.shape} does not match {self.check.shape[1]} columns.")
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite OSD probabilities: {np.count_nonzero(~np.isfinite(probabilities))} values.")
        if ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError(f"OSD probabilities outside [0,1]: min={probabilities.min():.17g}, max={probabilities.max():.17g}.")
        hard = ((probabilities > 0.5).astype(np.uint8) if self._hard_centered
                else np.zeros(self.check.shape[1], dtype=np.uint8))
        residual = syndrome.astype(np.uint8) ^ ((self.check @ hard) % 2)
        if not residual.any():
            return hard
        if self._hard_centered:
            clipped = np.clip(probabilities, 1e-12, 1 - 1e-12)
            reliability = np.abs(np.log1p(-clipped) - np.log(clipped))
            order = np.argsort(reliability, kind="stable")
        else:
            # Descending error probability = ascending signed log((1-q)/q).
            # Sorting q directly also handles exact 0/1 without clipping ties.
            order = np.argsort(-probabilities, kind="stable")
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


class SyndromeOSD0(DirectOSD0):
    """Syndrome OSD-0 of Roffe et al., arXiv:2005.07016, section V.

    Select independent columns from highest to lowest error probability,
    set nonbasis bits to zero, and solve H e = s. No BP or higher-order
    search runs. Unlike DirectOSD0, this does not preserve hard decisions;
    callers implement the 'only invalid sectors' fallback policy.
    Equal probabilities use stable original-column order. This implements
    the published algorithm, not a bitwise clone of a specific ldpc release.
    """

    _hard_centered = False


__all__ = ["DirectOSD0", "SyndromeOSD0"]
