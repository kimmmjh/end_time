"""Bivariate-bicycle (BB) code specifications for neural decoders.

The two named codes in this module use the construction from Bravyi et al.,
``H_X = [A | B]`` and ``H_Z = [B.T | A.T]``, with

``A = x^3 + y + y^2`` and ``B = y^3 + x + x^2``.

The matrices are intentionally dense: the largest built-in code is only
``[[144, 12, 12]]`` and dense arrays make the GF(2) validation and PyTorch
buffer registration straightforward.  Tanner edges remain sparse and carry a
cyclic-shift orbit id.  Sharing neural parameters within each orbit is the BB
analogue of sharing convolution kernels under translations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

BinaryArray = NDArray[np.uint8]
IntegerArray = NDArray[np.int64]


def _gf2_rref(matrix: NDArray[np.generic]) -> tuple[BinaryArray, list[int]]:
    """Return reduced row-echelon form and pivot columns over GF(2)."""

    reduced = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, columns = reduced.shape
    pivot_columns: list[int] = []
    pivot_row = 0

    for column in range(columns):
        candidates = np.flatnonzero(reduced[pivot_row:, column])
        if candidates.size == 0:
            continue

        selected = pivot_row + int(candidates[0])
        if selected != pivot_row:
            reduced[[pivot_row, selected]] = reduced[[selected, pivot_row]]

        other_rows = np.flatnonzero(reduced[:, column])
        other_rows = other_rows[other_rows != pivot_row]
        if other_rows.size:
            reduced[other_rows] ^= reduced[pivot_row]

        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break

    return reduced, pivot_columns


def gf2_rank(matrix: NDArray[np.generic]) -> int:
    """Compute a matrix rank over GF(2)."""

    _, pivots = _gf2_rref(matrix)
    return len(pivots)


def _gf2_row_basis(matrix: NDArray[np.generic]) -> BinaryArray:
    reduced, pivots = _gf2_rref(matrix)
    return reduced[: len(pivots)].copy()


def _gf2_nullspace(matrix: NDArray[np.generic]) -> BinaryArray:
    """Return a row basis for the right nullspace over GF(2)."""

    reduced, pivots = _gf2_rref(matrix)
    columns = reduced.shape[1]
    pivot_set = set(pivots)
    free_columns = [column for column in range(columns) if column not in pivot_set]
    basis = np.zeros((len(free_columns), columns), dtype=np.uint8)

    for basis_row, free_column in enumerate(free_columns):
        basis[basis_row, free_column] = 1
        for row, pivot_column in enumerate(pivots):
            basis[basis_row, pivot_column] = reduced[row, free_column]

    return basis


def _gf2_inverse(matrix: NDArray[np.generic]) -> BinaryArray:
    square = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, columns = square.shape
    if rows != columns:
        raise ValueError("Only square GF(2) matrices can be inverted.")

    augmented = np.concatenate((square, np.eye(rows, dtype=np.uint8)), axis=1)
    for column in range(columns):
        candidates = np.flatnonzero(augmented[column:, column])
        if candidates.size == 0:
            raise ValueError("Matrix is singular over GF(2).")
        selected = column + int(candidates[0])
        if selected != column:
            augmented[[column, selected]] = augmented[[selected, column]]

        other_rows = np.flatnonzero(augmented[:, column])
        other_rows = other_rows[other_rows != column]
        if other_rows.size:
            augmented[other_rows] ^= augmented[column]

    return augmented[:, columns:]


def _quotient_basis(
    *, kernel_of: BinaryArray, modulo_rows: BinaryArray, expected_size: int
) -> BinaryArray:
    """Choose representatives of ``ker(kernel_of) / row(modulo_rows)``."""

    span = _gf2_row_basis(modulo_rows)
    current_rank = span.shape[0]
    representatives: list[BinaryArray] = []

    for candidate in _gf2_nullspace(kernel_of):
        trial = np.vstack((span, candidate))
        trial_rank = gf2_rank(trial)
        if trial_rank == current_rank:
            continue
        representatives.append(candidate.copy())
        span = trial
        current_rank = trial_rank
        if len(representatives) == expected_size:
            break

    if len(representatives) != expected_size:
        raise ValueError(
            "Could not construct the expected number of logical operators: "
            f"expected {expected_size}, found {len(representatives)}."
        )
    return np.asarray(representatives, dtype=np.uint8)


def _shift_matrix(ell: int, m: int, dx: int, dy: int) -> BinaryArray:
    """Matrix whose row ``(i,j)`` points to column ``(i+dx,j+dy)``."""

    cells = ell * m
    result = np.zeros((cells, cells), dtype=np.uint8)
    for i in range(ell):
        for j in range(m):
            row = i * m + j
            column = ((i + dx) % ell) * m + ((j + dy) % m)
            result[row, column] = 1
    return result


@dataclass(frozen=True)
class BBCodeSpec:
    """A validated CSS BB code and its equivariant Tanner-graph metadata.

    Check nodes are ordered as all rows of ``hx`` followed by all rows of
    ``hz``.  Qubits are ordered as the left block followed by the right block.
    Edges are sorted by ``(check_index, qubit_index)`` (row-major nonzeros of
    the vertically stacked check matrices).

    ``edge_orbit`` has twelve values: three polynomial terms for each of the
    pairs ``(X check, left)``, ``(X check, right)``, ``(Z check, left)``, and
    ``(Z check, right)``.  A simultaneous cyclic translation of checks and
    qubits never changes an edge's orbit.
    """

    name: str
    ell: int
    m: int
    n: int
    k: int
    d: int
    hx: BinaryArray
    hz: BinaryArray
    logicals_x: BinaryArray
    logicals_z: BinaryArray
    edge_index: IntegerArray
    edge_check_index: IntegerArray
    edge_qubit_index: IntegerArray
    edge_orbit: IntegerArray
    edge_check_type: IntegerArray
    edge_qubit_block: IntegerArray
    edge_term: IntegerArray
    edge_displacement: IntegerArray

    @property
    def cells(self) -> int:
        return self.ell * self.m

    @property
    def num_x_checks(self) -> int:
        return self.hx.shape[0]

    @property
    def num_z_checks(self) -> int:
        return self.hz.shape[0]

    @property
    def num_checks(self) -> int:
        return self.num_x_checks + self.num_z_checks

    @property
    def num_nodes(self) -> int:
        return self.num_checks + self.n

    @property
    def num_edges(self) -> int:
        return self.edge_check_index.size

    @property
    def num_edge_orbits(self) -> int:
        return 12

    @property
    def logicals(self) -> BinaryArray:
        """Canonical logical operators in binary symplectic ``[X | Z]`` form."""

        zeros = np.zeros_like(self.logicals_x)
        return np.concatenate(
            (
                np.concatenate((self.logicals_x, zeros), axis=1),
                np.concatenate((zeros, self.logicals_z), axis=1),
            ),
            axis=0,
        )

    @property
    def stabilizers(self) -> BinaryArray:
        """CSS stabilizers in binary symplectic ``[X | Z]`` form."""

        x_zeros = np.zeros_like(self.hx)
        z_zeros = np.zeros_like(self.hz)
        return np.concatenate(
            (
                np.concatenate((self.hx, x_zeros), axis=1),
                np.concatenate((z_zeros, self.hz), axis=1),
            ),
            axis=0,
        )

    @classmethod
    def bb72(cls) -> "BBCodeSpec":
        """Return the published ``[[72, 12, 6]]`` BB code."""

        return cls._build(name="bb72", ell=6, m=6, expected_k=12, distance=6)

    @classmethod
    def bb144(cls) -> "BBCodeSpec":
        """Return the published ``[[144, 12, 12]]`` (gross) BB code."""

        return cls._build(name="bb144", ell=12, m=6, expected_k=12, distance=12)

    @classmethod
    def from_name(cls, name: str | int) -> "BBCodeSpec":
        normalized = str(name).lower().replace(" ", "")
        if normalized in {"72", "bb72", "[[72,12,6]]"}:
            return cls.bb72()
        if normalized in {"144", "bb144", "gross", "[[144,12,12]]"}:
            return cls.bb144()
        raise ValueError(f"Unknown BB code {name!r}; choose 'bb72' or 'bb144'.")

    @classmethod
    def _build(
        cls, *, name: str, ell: int, m: int, expected_k: int, distance: int
    ) -> "BBCodeSpec":
        cells = ell * m
        a_terms = ((3, 0), (0, 1), (0, 2))
        b_terms = ((0, 3), (1, 0), (2, 0))

        a = np.zeros((cells, cells), dtype=np.uint8)
        b = np.zeros((cells, cells), dtype=np.uint8)
        for dx, dy in a_terms:
            a ^= _shift_matrix(ell, m, dx, dy)
        for dx, dy in b_terms:
            b ^= _shift_matrix(ell, m, dx, dy)

        hx = np.concatenate((a, b), axis=1)
        hz = np.concatenate((b.T, a.T), axis=1)
        n = 2 * cells
        k = n - gf2_rank(hx) - gf2_rank(hz)
        if k != expected_k:
            raise ValueError(
                f"{name} construction produced k={k}, expected {expected_k}."
            )
        if np.any((hx @ hz.T) % 2):
            raise ValueError(f"{name} has non-commuting CSS checks.")

        logicals_x = _quotient_basis(kernel_of=hz, modulo_rows=hx, expected_size=k)
        logicals_z = _quotient_basis(kernel_of=hx, modulo_rows=hz, expected_size=k)
        pairing = (logicals_x @ logicals_z.T) % 2
        transform = _gf2_inverse(pairing).T
        logicals_z = (transform @ logicals_z) % 2

        edge_metadata = cls._make_edge_metadata(
            ell=ell,
            m=m,
            a_terms=a_terms,
            b_terms=b_terms,
            hx=hx,
            hz=hz,
        )
        return cls(
            name=name,
            ell=ell,
            m=m,
            n=n,
            k=k,
            d=distance,
            hx=hx,
            hz=hz,
            logicals_x=logicals_x,
            logicals_z=logicals_z,
            **edge_metadata,
        )

    @staticmethod
    def _make_edge_metadata(
        *,
        ell: int,
        m: int,
        a_terms: tuple[tuple[int, int], ...],
        b_terms: tuple[tuple[int, int], ...],
        hx: BinaryArray,
        hz: BinaryArray,
    ) -> dict[str, IntegerArray]:
        cells = ell * m
        check_indices: list[int] = []
        qubit_indices: list[int] = []
        orbits: list[int] = []
        check_types: list[int] = []
        qubit_blocks: list[int] = []
        terms: list[int] = []
        displacements: list[tuple[int, int]] = []

        groups = (
            # check type, qubit block, polynomial terms, transpose, orbit base
            (0, 0, a_terms, False, 0),
            (0, 1, b_terms, False, 3),
            (1, 0, b_terms, True, 6),
            (1, 1, a_terms, True, 9),
        )
        for check_type, block, polynomial, transpose, orbit_base in groups:
            for check_cell in range(cells):
                i, j = divmod(check_cell, m)
                for term_index, (raw_dx, raw_dy) in enumerate(polynomial):
                    dx = -raw_dx if transpose else raw_dx
                    dy = -raw_dy if transpose else raw_dy
                    qubit_cell = ((i + dx) % ell) * m + ((j + dy) % m)
                    check_indices.append(check_type * cells + check_cell)
                    qubit_indices.append(block * cells + qubit_cell)
                    orbits.append(orbit_base + term_index)
                    check_types.append(check_type)
                    qubit_blocks.append(block)
                    terms.append(term_index)
                    displacements.append((dx, dy))

        check_array = np.asarray(check_indices, dtype=np.int64)
        qubit_array = np.asarray(qubit_indices, dtype=np.int64)
        order = np.lexsort((qubit_array, check_array))
        check_array = check_array[order]
        qubit_array = qubit_array[order]

        stacked = np.concatenate((hx, hz), axis=0)
        expected_check, expected_qubit = np.nonzero(stacked)
        if not (
            np.array_equal(check_array, expected_check)
            and np.array_equal(qubit_array, expected_qubit)
        ):
            raise ValueError("Tanner edge metadata does not align with Hx/Hz.")

        num_checks = 2 * cells
        return {
            "edge_index": np.stack((check_array, num_checks + qubit_array), axis=0),
            "edge_check_index": check_array,
            "edge_qubit_index": qubit_array,
            "edge_orbit": np.asarray(orbits, dtype=np.int64)[order],
            "edge_check_type": np.asarray(check_types, dtype=np.int64)[order],
            "edge_qubit_block": np.asarray(qubit_blocks, dtype=np.int64)[order],
            "edge_term": np.asarray(terms, dtype=np.int64)[order],
            "edge_displacement": np.asarray(displacements, dtype=np.int64)[order],
        }

    def translation_permutations(self, dx: int, dy: int) -> dict[str, IntegerArray]:
        """Return old-to-new index maps for a simultaneous cyclic translation.

        For example, ``translated[:, permutation] = original`` moves a feature
        array using the returned old-to-new convention.  The edge permutation
        preserves ``edge_orbit`` exactly.
        """

        cell_permutation = np.empty(self.cells, dtype=np.int64)
        for cell in range(self.cells):
            i, j = divmod(cell, self.m)
            cell_permutation[cell] = ((i + dx) % self.ell) * self.m + (
                (j + dy) % self.m
            )

        check_permutation = np.concatenate(
            (cell_permutation, self.cells + cell_permutation)
        )
        qubit_permutation = np.concatenate(
            (cell_permutation, self.cells + cell_permutation)
        )
        node_permutation = np.concatenate(
            (check_permutation, self.num_checks + qubit_permutation)
        )

        lookup = {
            (int(check), int(qubit)): edge
            for edge, (check, qubit) in enumerate(
                zip(self.edge_check_index, self.edge_qubit_index)
            )
        }
        edge_permutation = np.asarray(
            [
                lookup[
                    (
                        int(check_permutation[check]),
                        int(qubit_permutation[qubit]),
                    )
                ]
                for check, qubit in zip(self.edge_check_index, self.edge_qubit_index)
            ],
            dtype=np.int64,
        )
        if not np.array_equal(self.edge_orbit, self.edge_orbit[edge_permutation]):
            raise ValueError("Cyclic translation unexpectedly changed an edge orbit.")

        return {
            "cells": cell_permutation,
            "checks": check_permutation,
            "qubits": qubit_permutation,
            "nodes": node_permutation,
            "edges": edge_permutation,
        }

    def torch_buffers(
        self, *, device: Any = None, matrix_dtype: Any = None
    ) -> dict[str, Any]:
        """Create tensors suitable for ``register_buffer`` on a neural model."""

        import torch

        if matrix_dtype is None:
            matrix_dtype = torch.float32
        matrices = {
            "hx": self.hx,
            "hz": self.hz,
            "logicals_x": self.logicals_x,
            "logicals_z": self.logicals_z,
        }
        result = {
            name: torch.as_tensor(value, dtype=matrix_dtype, device=device)
            for name, value in matrices.items()
        }
        for name in (
            "edge_index",
            "edge_check_index",
            "edge_qubit_index",
            "edge_orbit",
            "edge_check_type",
            "edge_qubit_block",
            "edge_term",
            "edge_displacement",
        ):
            result[name] = torch.as_tensor(
                getattr(self, name), dtype=torch.long, device=device
            )
        return result
