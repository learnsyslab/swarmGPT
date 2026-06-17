"""First-class Bernstein (Bézier) spline type for SwarmGPT 2 trajectories.

Every spline is a Bernstein polynomial stored as control points over a real time
interval ``[t0, t1]``. Operations act on control points so the curve stays exactly
in the Bernstein basis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

SplineDict = dict[int, "Spline"]
"""Mapping of 0-indexed drone id to its intended trajectory spline (spline 1)."""


class SplinePrimitive(Protocol):
    """The WS1 primitive contract: a callable returning one ``Spline`` per drone.

    Replaces the legacy ``primitive(...) -> (final_pos, {time: {drone: pos}})`` waypoint
    signature. Existing primitives are migrated to this contract in WS1, behind the
    ``use_motion_primitives`` flag; this Protocol only declares the shape.
    """

    def __call__(self, *args: object, **kwargs: object) -> SplineDict:
        """Evaluate the primitive into per-drone splines over its block interval."""
        ...


def _bezier_product_1d(p: NDArray, q: NDArray) -> NDArray:
    """Multiply two 1-D Bézier control-point vectors, returning the product's control points.

    For control points ``p`` (degree ``m``) and ``q`` (degree ``d``) the product is degree
    ``m + d`` with ``f[k] = sum_j C(m,j) C(d,k-j) p[j] q[k-j] / C(m+d,k)``.

    Args:
        p: Control points of the first factor, shape ``(m + 1,)``.
        q: Control points of the second factor, shape ``(d + 1,)``.

    Returns:
        Control points of the product, shape ``(m + d + 1,)``.
    """
    m = len(p) - 1
    d = len(q) - 1
    out = np.zeros(m + d + 1)
    for k in range(m + d + 1):
        total = 0.0
        for j in range(max(0, k - d), min(m, k) + 1):
            total += math.comb(m, j) * math.comb(d, k - j) * p[j] * q[k - j]
        out[k] = total / math.comb(m + d, k)
    return out


class Spline:
    """A single-segment Bernstein (Bézier) curve in 1-D or 3-D over ``[t0, t1]``.

    Attributes:
        control_points: Float array of shape ``(degree + 1, dim)``.
        t0: Interval start time in seconds.
        t1: Interval end time in seconds.
    """

    def __init__(self, control_points: NDArray, t0: float = 0.0, t1: float = 1.0) -> None:
        """Build a spline from control points and a time interval.

        Args:
            control_points: Array of shape ``(degree + 1, dim)`` with ``dim`` typically 1 for
                scalar splines or 3 for spatial.
            t0: Interval start time in seconds.
            t1: Interval end time in seconds. Must be strictly greater than ``t0``.

        Raises:
            ValueError: If the interval is empty or the control-point shape is invalid.
        """
        cp = np.asarray(control_points, dtype=float)
        if cp.ndim != 2 or cp.shape[0] < 1:
            raise ValueError(f"control_points must be (degree+1, dim), got shape {cp.shape}")
        if t1 <= t0:
            raise ValueError(f"Spline interval must satisfy t1 > t0, got [{t0}, {t1}]")
        self.control_points = cp
        self.t0 = float(t0)
        self.t1 = float(t1)

    @property
    def degree(self) -> int:
        """Polynomial degree of the spline (one less than the control-point count)."""
        return self.control_points.shape[0] - 1

    @property
    def dim(self) -> int:
        """Spatial dimension of the spline (1 for scalar, 3 for spatial)."""
        return self.control_points.shape[1]

    @property
    def duration(self) -> float:
        """Length of the time interval in seconds."""
        return self.t1 - self.t0

    def evaluate(self, t: float | NDArray) -> NDArray:
        """Evaluate the curve at one time or an array of times.

        Args:
            t: A scalar time or a 1-D array of times in seconds. Clamped to ``[t0, t1]``.

        Returns:
            Shape ``(dim,)`` for scalar ``t``; shape ``(len(t), dim)`` for an array.
        """
        scalar_input = np.ndim(t) == 0
        u = (np.atleast_1d(np.asarray(t, dtype=float)) - self.t0) / self.duration
        u = np.clip(u, 0.0, 1.0)
        n = self.degree
        i = np.arange(n + 1)
        binom = np.array([math.comb(n, k) for k in range(n + 1)], dtype=float)
        basis = binom[None, :] * u[:, None] ** i[None, :] * (1.0 - u)[:, None] ** (n - i)[None, :]
        result = basis @ self.control_points
        return result[0] if scalar_input else result

    def derivative(self) -> Spline:
        """Return the time-derivative curve (degree ``n-1``) over the same interval.

        The Bézier derivative has control points ``n * (c[i+1] - c[i]) / duration``;
        a degree-0 spline differentiates to a constant-zero spline.

        Returns:
            A new ``Spline`` representing d/dt of this curve.
        """
        n = self.degree
        if n == 0:
            return Spline(np.zeros((1, self.dim)), self.t0, self.t1)
        diff = self.control_points[1:] - self.control_points[:-1]
        return Spline(n * diff / self.duration, self.t0, self.t1)

    def _velocity_acceleration(self) -> tuple[Spline, Spline]:
        """Return the first and second derivative splines.

        Returns:
            A tuple ``(velocity, acceleration)`` as ``Spline`` objects.
        """
        vel = self.derivative()
        acc = vel.derivative()
        return vel, acc

    def start_state(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(position, velocity, acceleration)`` at ``t0``, read from control points.

        Returns:
            Three arrays of shape ``(dim,)``. No curve evaluation is performed.
        """
        vel, acc = self._velocity_acceleration()
        return self.control_points[0], vel.control_points[0], acc.control_points[0]

    def end_state(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(position, velocity, acceleration)`` at ``t1``, read from control points.

        Returns:
            Three arrays of shape ``(dim,)``. No curve evaluation is performed.
        """
        vel, acc = self._velocity_acceleration()
        return self.control_points[-1], vel.control_points[-1], acc.control_points[-1]

    def degree_elevate(self, to_degree: int) -> Spline:
        """Return an equal curve expressed at a higher (or equal) degree.

        Uses the standard single-step Bézier elevation
        ``c'[i] = (i/m) c[i-1] + (1 - i/m) c[i]`` repeatedly until ``to_degree`` is reached.

        Args:
            to_degree: Target degree. Must be ``>= self.degree``.

        Returns:
            A new ``Spline`` of degree ``to_degree`` describing the same curve.

        Raises:
            ValueError: If ``to_degree`` is less than the current degree.
        """
        if to_degree < self.degree:
            raise ValueError(f"Cannot elevate degree {self.degree} down to {to_degree}")
        cp = self.control_points
        n = self.degree
        while n < to_degree:
            m = n + 1
            new = np.zeros((m + 1, self.dim))
            new[0] = cp[0]
            new[m] = cp[n]
            for i in range(1, m):
                alpha = i / m
                new[i] = alpha * cp[i - 1] + (1.0 - alpha) * cp[i]
            cp, n = new, m
        return Spline(cp, self.t0, self.t1)

    def __add__(self, other: Spline) -> Spline:
        """Add two splines over the same interval, elevating to the common degree.

        Args:
            other: A ``Spline`` with the same interval and dimension.

        Returns:
            A new ``Spline`` equal to the pointwise sum.

        Raises:
            ValueError: If intervals or dimensions differ.
        """
        if (self.t0, self.t1) != (other.t0, other.t1):
            raise ValueError("Cannot add splines over different intervals")
        if self.dim != other.dim:
            raise ValueError(f"Cannot add splines of dim {self.dim} and {other.dim}")
        n = max(self.degree, other.degree)
        left = self.degree_elevate(n)
        right = other.degree_elevate(n)
        return Spline(left.control_points + right.control_points, self.t0, self.t1)

    def __mul__(self, other: float | Spline) -> Spline:
        """Multiply by a scalar, another scalar spline, or scale a vector spline.

        Supported: ``float`` (scales all control points); scalar×scalar (1-D product);
        scalar×vector or vector×scalar (scales each axis by the time-varying scalar).

        Args:
            other: A number, or a ``Spline`` over the same interval.

        Returns:
            A new ``Spline``. Product degree is the sum of factor degrees.

        Raises:
            ValueError: For mismatched intervals, or vector×vector multiplication.
        """
        if isinstance(other, (int, float)):
            return Spline(self.control_points * float(other), self.t0, self.t1)
        if (self.t0, self.t1) != (other.t0, other.t1):
            raise ValueError("Cannot multiply splines over different intervals")
        if self.dim == 1 and other.dim == 1:
            product = _bezier_product_1d(self.control_points[:, 0], other.control_points[:, 0])
            return Spline(product[:, None], self.t0, self.t1)
        if self.dim == 1:
            scalar, vector = self, other
        elif other.dim == 1:
            scalar, vector = other, self
        else:
            raise ValueError("Spline product requires a scalar (dim=1) factor")
        cols = [
            _bezier_product_1d(scalar.control_points[:, 0], vector.control_points[:, axis])
            for axis in range(vector.dim)
        ]
        return Spline(np.stack(cols, axis=1), self.t0, self.t1)

    __rmul__ = __mul__

    def affine_transform(self, matrix: NDArray) -> Spline:
        """Apply a 4x4 homogeneous transform to the curve via its control points.

        Bernstein curves are affine-invariant, so transforming the control points
        transforms the whole curve. Used to place precomputed canonical arcs (WS1).

        Args:
            matrix: A ``(4, 4)`` homogeneous transform.

        Returns:
            A new 3-D ``Spline`` with each control point transformed.

        Raises:
            ValueError: If the spline is not 3-D or ``matrix`` is not ``(4, 4)``.
        """
        matrix = np.asarray(matrix, dtype=float)
        if self.dim != 3:
            raise ValueError("affine_transform requires a 3-D spline")
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix must be (4, 4), got {matrix.shape}")
        homogeneous = np.hstack([self.control_points, np.ones((self.degree + 1, 1))])
        transformed = (matrix @ homogeneous.T).T[:, :3]
        return Spline(transformed, self.t0, self.t1)

    def axis_bounds(self) -> tuple[NDArray, NDArray]:
        """Return per-axis ``(lower, upper)`` bounds from the control-point hull.

        By convex-hull containment the curve lies within this box for all ``t``,
        giving a conservative spatial bound without evaluation (used by the safety filter).

        Returns:
            Two arrays of shape ``(dim,)``: the per-axis minimum and maximum control values.
        """
        return self.control_points.min(axis=0), self.control_points.max(axis=0)

    def to_waypoints(self, freq: float) -> dict[str, NDArray]:
        """Sample the spline at ``freq`` Hz into a ``{time, pos, vel, acc}`` dict.

        This is the only sampling step in the pipeline (spec §A.5): the read-out of a
        safe reference spline at control frequency.

        Args:
            freq: Sampling frequency in Hz.

        Returns:
            Dict with ``time`` shape ``(T,)`` and ``pos``/``vel``/``acc`` shape ``(T, dim)``.
        """
        n_samples = int(round(self.duration * freq)) + 1
        times = np.linspace(self.t0, self.t1, n_samples)
        vel, acc = self._velocity_acceleration()
        return {
            "time": times,
            "pos": self.evaluate(times),
            "vel": vel.evaluate(times),
            "acc": acc.evaluate(times),
        }

    @classmethod
    def from_axswarm_zeta(
        cls, zeta: NDArray, degree: int, t0: float = 0.0, t1: float = 1.0
    ) -> Spline:
        """Build a 3-D spline from an axswarm ``zeta`` control-point vector.

        axswarm lays out ``zeta`` as ``[x_0..x_N, y_0..y_N, z_0..z_N]`` (see
        ``axswarm/spline.py`` ``_expand_coeff``), where ``N`` is the spline degree.

        Args:
            zeta: Flat control-point vector of length ``3 * (degree + 1)``.
            degree: Bernstein degree ``N`` used by the solver.
            t0: Interval start time in seconds.
            t1: Interval end time in seconds.

        Returns:
            A 3-D ``Spline`` with control points of shape ``(degree + 1, 3)``.

        Raises:
            ValueError: If ``zeta`` does not have length ``3 * (degree + 1)``.
        """
        expected = 3 * (degree + 1)
        zeta = np.asarray(zeta, dtype=float)
        if zeta.shape != (expected,):
            raise ValueError(f"zeta must have length {expected}, got {zeta.shape}")
        control_points = zeta.reshape(3, degree + 1).T
        return cls(control_points, t0, t1)


class PiecewiseSpline:
    """An ordered set of contiguous ``Spline`` segments forming one continuous curve.

    Attributes:
        segments: The component splines, with chained intervals ``[t0, t1]``.
    """

    def __init__(self, segments: list[Spline]) -> None:
        """Build a piecewise spline from contiguous segments.

        Args:
            segments: Non-empty list of splines whose intervals join end-to-end.

        Raises:
            ValueError: If the list is empty or adjacent intervals are not contiguous.
        """
        if not segments:
            raise ValueError("PiecewiseSpline needs at least one segment")
        for left, right in zip(segments[:-1], segments[1:]):
            if not math.isclose(left.t1, right.t0, abs_tol=1e-9):
                raise ValueError(f"Segments must be contiguous: {left.t1} != {right.t0}")
        self.segments = segments
        self._boundaries = np.array([seg.t1 for seg in segments])

    @property
    def t0(self) -> float:
        """Start time of the whole curve."""
        return self.segments[0].t0

    @property
    def t1(self) -> float:
        """End time of the whole curve."""
        return self.segments[-1].t1

    def _segment_at(self, t: float) -> Spline:
        """Return the segment whose interval contains ``t``.

        Args:
            t: A time value in seconds.

        Returns:
            The first segment whose ``t1`` is >= ``t`` (with tolerance), or the last segment.
        """
        for segment in self.segments:
            if t <= segment.t1 + 1e-9:
                return segment
        return self.segments[-1]

    def evaluate(self, t: float | NDArray) -> NDArray:
        """Evaluate the curve at one time or an array of times.

        Args:
            t: A scalar time or 1-D array of times in seconds.

        Returns:
            Shape ``(dim,)`` for scalar ``t``; shape ``(len(t), dim)`` for an array.
        """
        if np.ndim(t) == 0:
            return self._segment_at(float(t)).evaluate(t)
        t_arr = np.asarray(t, dtype=float)
        indices = np.searchsorted(self._boundaries, t_arr, side="left")
        indices = np.clip(indices, 0, len(self.segments) - 1)
        dim = self.segments[0].dim
        result = np.empty((len(t_arr), dim))
        for idx in range(len(self.segments)):
            mask = indices == idx
            if mask.any():
                result[mask] = self.segments[idx].evaluate(t_arr[mask])
        return result

    def derivative(self) -> PiecewiseSpline:
        """Return the per-segment derivative as a new piecewise spline.

        Returns:
            A new ``PiecewiseSpline`` of per-segment derivatives.
        """
        return PiecewiseSpline([segment.derivative() for segment in self.segments])

    def start_state(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(position, velocity, acceleration)`` at the curve start.

        Returns:
            Three arrays of shape ``(dim,)``.
        """
        return self.segments[0].start_state()

    def end_state(self) -> tuple[NDArray, NDArray, NDArray]:
        """Return ``(position, velocity, acceleration)`` at the curve end.

        Returns:
            Three arrays of shape ``(dim,)``.
        """
        return self.segments[-1].end_state()
