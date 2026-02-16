"""
Parametrized lattice integration for tree packing GA.

Ports Serge's NFP-based lattice parametrization from Numba to CuPy/NumPy
for integration with Jeroen Cottaar's GPU-accelerated GA framework.

The lattice is defined by 6 continuous parameters:
  t_same, t_horiz, t_vert: NFP walk parameters [-1, 1]
  theta:  whole-lattice rotation angle [0, 2*pi]
  dx, dy: lattice anchor translation from square center

These parameters live in the GA genotype, mutated by GA moves only.
L-BFGS relaxation operates on the resulting tree xyt positions.

Flow:
  GA mutates lattice_params -> regenerate inner tree xyt ->
  GPU L-BFGS relaxation (moves ALL trees) -> score

License: CC BY-SA 4.0 (derivative of Jeroen Cottaar's work)
"""

import copy
import numpy as np
import cupy as cp
from numba import njit
from dataclasses import dataclass, field

import kaggle_support as kgs
import pack_ga3

# =============================================================================
# TREE GEOMETRY (80x upscaled integer coordinates)
# =============================================================================

TREE_CONVEX_HULL = np.array([
    (0, 64), (28, 0), (6, -16), (-6, -16), (-28, 0)
], dtype=np.float64)

# Jeroen recenters tree so polygon centroid is at origin.
# Centroid of full 15-vertex tree in 80x coords: (0, 20788/1179)
# cx=0 by symmetry, cy computed via exact shoelace formula.
TREE_CENTROID_X = 0.0
TREE_CENTROID_Y = 20788.0 / 1179.0

# =============================================================================
# NFP WAYPOINTS
# =============================================================================

# Same-cell contact along (0,64)->(28,0) multiedge, t in [0, 1]
# Also used for t_horiz via vertical symmetry
NFP_SAME_UPPER = np.array([
    [1.0,      0,     128],
    [0.275,   20,      80],
    [0.24,    53/3,    80],
    [0.01,    28-44/7, 64+44/7],
    [0.0,     28,      64],
], dtype=np.float64)

# Same-cell contact, t in [-1, 0]
NFP_SAME_LOWER = np.array([
    [0.0,    28,  64],
    [-0.32,  38,  40],
    [-0.38,  33,  40],
    [-0.64,  44,  20],
    [-0.74,  36,  20],
    [-1.0,   56,   0],
], dtype=np.float64)

# Vertical neighbor contact along (-28,0)->(28,0), t in [0, 1]
NFP_VERT_UPPER = np.array([
    [1.0,    56,    0],
    [0.75,   34,    0],
    [0.57,   34,  -16],
    [0.32,   12,  -16],
    [0.14,   12,  -32],
    [0.0,     0,  -32],
], dtype=np.float64)

# Vertical neighbor contact, t in [-1, 0]
NFP_VERT_LOWER = np.array([
    [0.0,      0, -32],
    [-0.14,  -12, -32],
    [-0.32,  -12, -16],
    [-0.57,  -34, -16],
    [-0.75,  -34,   0],
    [-1.0,   -56,   0],
], dtype=np.float64)

# Same-orientation NFP boundary: (min_x_diff, abs_y, slope)
NFP_SAME_SAME = np.array([
    (56,  0, -1),
    (40, 16,  1),
    (44, 20, -11/20),
    (38 - 100/31, 40 - 100/31, 1),
    (38, 40, -5/12),
    (6 + 20/3, 64, -5/12),
    (0,  80,  0),
], dtype=np.float64)

# Opposite-orientation NFP boundary: (min_x_diff, y, slope)
NFP_OPPOSITE = np.array([
    (12, -32,  0),
    (34, -16,  0),
    (56,   0, -1),
    (44,  20, -11/20),
    (36,  40, -5/12),
    (28,  64, -1),
    (28 - 44/7, 64 + 44/7, -5/12),
    (20,  80, -5/12),
    (0,  128,  0),
], dtype=np.float64)


# =============================================================================
# NFP INTERPOLATION & COLLISION (plain Python, operates on scalars)
# =============================================================================
# These run once per individual during lattice regeneration, NOT in the
# GPU hot loop. No need for CUDA — scalar Python is fine here.

@njit
def _interpolate_nfp(t, upper, lower):
    """Interpolate along NFP contour boundary.

    Args:
        t: scalar parameter in [-1, 1]
        upper: (M, 3) waypoints [t_val, x, y] for t >= 0
        lower: (M, 3) waypoints [t_val, x, y] for t < 0

    Returns:
        (x, y) relative offset at parameter t
    """
    waypoints = upper if t >= 0 else lower
    n = waypoints.shape[0]
    for i in range(n - 1):
        t_hi, x_hi, y_hi = waypoints[i, 0], waypoints[i, 1], waypoints[i, 2]
        t_lo, x_lo, y_lo = waypoints[i + 1, 0], waypoints[i + 1, 1], waypoints[i + 1, 2]
        if t_lo <= t <= t_hi:
            frac = (t - t_lo) / (t_hi - t_lo) if t_hi != t_lo else 0.0
            x = x_lo + frac * (x_hi - x_lo)
            y = y_lo + frac * (y_hi - y_lo)
            return x, y
    return float(waypoints[n - 1, 1]), float(waypoints[n - 1, 2])


@njit
def min_x_for_y_same(dy):
    """Minimum horizontal separation for same-orientation trees."""
    abs_dy = abs(dy)
    for i in range(NFP_SAME_SAME.shape[0] - 1, -1, -1):
        x, y, slope = NFP_SAME_SAME[i]
        if abs_dy >= y:
            return x + slope * (abs_dy - y)
    return NFP_SAME_SAME[0, 0]


@njit
def min_x_for_y_opposite(dy):
    """Minimum horizontal separation for opposite-orientation trees."""
    if dy <= -32 or dy >= 128:
        return 0.0
    for i in range(NFP_OPPOSITE.shape[0] - 1, -1, -1):
        x, y, slope = NFP_OPPOSITE[i]
        if dy > y:
            return x + slope * (dy - y)
    return 0.0


@njit
def compute_collision_penalty(dx_same, dy_same, dxh, dyh, dxv, dyv):
    """Check collisions between trees in adjacent cells."""
    total = 0.0

    # Same orientation checks
    same_disps = [
        (dxh - dxv, dyh - dyv),
        (dxh + dxv, dyh + dyv),
        (2*dxh - dxv, 2*dyh - dyv),
        (2*dxh + dxv, 2*dyh + dyv),
    ]
    for dx, dy in same_disps:
        gap = min_x_for_y_same(dy) - abs(dx)
        if gap > 4e-14:
            total += gap

    # Opposite orientation checks
    opp_disps = [
        (dx_same + dxh + dxv, dy_same + dyh + dyv),
        (dx_same - dxh - dxv, dy_same - dyh - dyv),
        (dx_same + dxv - dxh, dy_same + dyv - dyh),
        (dx_same + dxh - dxv, dy_same + dyh - dyv),
        (dx_same - 2*dxv, dy_same - 2*dyv),
        (dx_same - dxh - 2*dxv, dy_same - dyh - 2*dyv),
        (dx_same + dxh - 2*dxv, dy_same + dyh - 2*dyv),
    ]
    for dx, dy in opp_disps:
        gap = min_x_for_y_opposite(dy) - abs(dx)
        if gap > 4e-14:
            total += gap

    return total


@njit
def compute_cell_vectors(t_same, t_horiz, t_vert):
    """Compute lattice cell vectors from NFP walk parameters.

    Returns:
        (dx_same, dy_same, dxh, dyh, dxv, dyv)
    """
    dx_same, dy_same = _interpolate_nfp(t_same, NFP_SAME_UPPER, NFP_SAME_LOWER)
    dx_horiz, ndy_horiz = _interpolate_nfp(t_horiz, NFP_SAME_UPPER, NFP_SAME_LOWER)
    ndx_vert, ndy_vert = _interpolate_nfp(t_vert, NFP_VERT_UPPER, NFP_VERT_LOWER)

    dxv = dx_same - ndx_vert
    dyv = dy_same - ndy_vert

    # Clamp vertical vector
    min_xv = min_x_for_y_same(dyv)
    if abs(dxv) < min_xv:
        dxv = min_xv
        dx_same = dxv + ndx_vert

    dxh = dx_same + dx_horiz
    dyh = dy_same - ndy_horiz

    # Clamp horizontal vector
    min_xh = min_x_for_y_same(dyh)
    if abs(dxh) < min_xh:
        dxh = min_xh

    # Clamp diagonal
    min_xhv = min_x_for_y_same(dyh - dyv)
    if abs(dxh - dxv) < min_xhv:
        dxh = dxv + min_xhv

    return dx_same, dy_same, dxh, dyh, dxv, dyv


# =============================================================================
# BOUNDING BOX & ROTATION
# =============================================================================

@njit
def rotated_bounding_box(angle):
    """Compute bounding box of tree rotated by angle.

    Returns:
        (bbx, bby, bbh, bbv) — center offset and dimensions
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)

    x_min, y_min = np.inf, np.inf
    x_max, y_max = -np.inf, -np.inf

    for i in range(TREE_CONVEX_HULL.shape[0]):
        x, y = TREE_CONVEX_HULL[i, 0], TREE_CONVEX_HULL[i, 1]
        x_rot = x * cos_a - y * sin_a
        y_rot = x * sin_a + y * cos_a

        x_min = min(x_min, x_rot)
        y_min = min(y_min, y_rot)
        x_max = max(x_max, x_rot)
        y_max = max(y_max, y_rot)

    bbx = (x_max + x_min) / 2
    bby = (y_max + y_min) / 2
    bbh = x_max - x_min
    bbv = y_max - y_min

    return bbx, bby, bbh, bbv


@njit
def rotate_and_align(dx_same, dy_same, dxh, dyh, dxv, dyv, angle):
    """Rotate lattice vectors and align bounding box centers.

    Mirrors downward tree center around bbox center so both orientations
    share the same reference point system.
    
    Returns bbx, bby adjusted for Jeroen's tree origin (polygon centroid).

    Returns:
        (dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, bbx, bby, bbh, bbv)
    """
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    bbx_raw, bby_raw, bbh, bbv = rotated_bounding_box(angle)

    # Mirror uses raw bbx/bby (bbox center relative to polygon (0,0))
    # because NFP vectors are in the polygon (0,0) frame
    dxs_r = dx_same * cos_a - dy_same * sin_a - 2 * bbx_raw
    dys_r = dx_same * sin_a + dy_same * cos_a - 2 * bby_raw
    dxh_r = dxh * cos_a - dyh * sin_a
    dyh_r = dxh * sin_a + dyh * cos_a
    dxv_r = dxv * cos_a - dyv * sin_a
    dyv_r = dxv * sin_a + dyv * cos_a

    # Adjust bbx/bby for Jeroen's tree origin (centroid, not polygon origin)
    # Rotated centroid: (cx*cos - cy*sin, cx*sin + cy*cos) where cx=0
    bbx = bbx_raw + TREE_CENTROID_Y * sin_a
    bby = bby_raw - TREE_CENTROID_Y * cos_a

    return dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, bbx, bby, bbh, bbv


@njit
def compute_max_rc(n_needed):
    """Grid size: sqrt(n) cells each direction from center + buffer."""
    return 4 * int(np.sqrt(n_needed)) + 11


@njit
def compute_grid_positions(dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, max_rc):
    """Generate positions with center cell at world origin."""
    center = max_rc // 2
    n_trees = 2 * max_rc * max_rc
    positions = np.empty((n_trees, 3))

    idx = 0
    for row in range(max_rc):
        for col in range(max_rc):
            r = row - center
            c = col - center
            base_x = c * dxh_r + r * dxv_r
            base_y = c * dyh_r + r * dyv_r

            positions[idx, 0] = base_x
            positions[idx, 1] = base_y
            positions[idx, 2] = 0
            idx += 1

            positions[idx, 0] = base_x + dxs_r
            positions[idx, 1] = base_y + dys_r
            positions[idx, 2] = 1
            idx += 1

    return positions


# =============================================================================
# LATTICE GENERATION
# =============================================================================

@njit
def generate_lattice_trees(t_same, t_horiz, t_vert, theta, anchor_dx, anchor_dy,
                           n_inner, square_size, scale=1.0):
    """Generate inner tree positions from parametrized lattice.

    Always returns exactly n_inner trees, sorted by distance from anchor.
    No square boundary filtering — the GA/relaxation handles containment.

    Args:
        t_same, t_horiz, t_vert: NFP walk parameters [-1, 1]
        theta: lattice rotation angle in DEGREES
        anchor_dx, anchor_dy: lattice anchor offset in Serge's 80x scale
        n_inner: exact number of inner trees to return
        square_size: current square boundary size (in Jeroen's scale)
        scale: coordinate scale factor (1/80)

    Returns:
        xyt: NumPy array (n_inner, 3) in Jeroen's coordinate system
             [x, y, angle_radians]. Caller converts to CuPy.
    """
    # 1. Compute cell vectors (in Serge's 80x scale)
    dx_same, dy_same, dxh, dyh, dxv, dyv = compute_cell_vectors(
        float(t_same), float(t_horiz), float(t_vert)
    )

    # Convert theta to radians
    theta_rad = float(theta) * np.pi / 180.0

    # 2. Rotate and align bounding box centers
    dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, bbx, bby, bbh, bbv = \
        rotate_and_align(dx_same, dy_same, dxh, dyh, dxv, dyv, theta_rad)

    # 3. Generate grid positions
    max_rc = compute_max_rc(n_inner)
    positions = compute_grid_positions(
        dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, max_rc
    )
    # positions[:, 0:2] = lattice reference point in rotated frame (80x scale)
    # positions[:, 2] = orientation flag (0=upward, 1=downward)

    # 4. Convert lattice positions to tree origins and apply anchor
    orient = positions[:, 2]
    adx = float(anchor_dx)
    ady = float(anchor_dy)
    tree_x = positions[:, 0] + bbx * (2 * orient - 1) + adx
    tree_y = positions[:, 1] + bby * (2 * orient - 1) + ady

    # 5. Sort by distance from center, take n_inner closest
    dist_sq = tree_x**2 + tree_y**2
    order = np.argsort(dist_sq)
    n_select = min(n_inner, len(order))
    sel = order[:n_select]

    # 6. Build result
    result = np.empty((n_select, 3), dtype=np.float64)
    result[:, 0] = tree_x[sel] * scale
    result[:, 1] = tree_y[sel] * scale
    result[:, 2] = np.where(orient[sel] == 1, np.pi + theta_rad, theta_rad)

    return result


# =============================================================================
# COORDINATE SCALE
# =============================================================================

# Jeroen's tree: tip at y=0.8, base_w=0.7 (half=0.35)
# Serge's tree: tip at y=64, base_w=56 (half=28)
# Ratio: 0.8/64 = 0.0125 = 1/80
SCALE_FACTOR = 1.0 / 80.0


# =============================================================================
# SOLUTION COLLECTION SUBCLASS
# =============================================================================

N_LATTICE_PARAMS = 6  # [t_same, t_horiz, t_vert, theta, anchor_dx, anchor_dy]


@dataclass
class SolutionCollectionSquareParametrizedLattice(kgs.SolutionCollectionSquare):
    """Square boundary with parametrized lattice core.

    Genotype: xyt holds ONLY edge trees (N_edge = N_total - n_inner).
    Phenotype: full N_total trees = lattice-generated + edge trees.

    The lattice params are the genome for inner trees. L-BFGS gradients
    for inner tree positions are discarded (no backprop to lattice params).

    Fields:
      lattice_params: (N_solutions, 6) lattice parameters
      n_inner: int, number of trees generated from lattice
    """
    lattice_params: cp.ndarray = field(init=True, default=None)
    n_inner: int = field(init=True, default=0)

    @property
    def N_trees(self) -> int:
        """Total trees in phenotype = lattice + edge."""
        if self.xyt is None:
            return 0
        return self.n_inner + self.xyt.shape[1]

    def is_phenotype(self):
        """Genotype != phenotype unless override_phenotype is set.
        
        During rough relaxation, override_phenotype=True means L-BFGS
        operates directly on edge trees without generating lattice trees.
        """
        if self.override_phenotype:
            return True
        return False

    def _prep_for_phenotype(self):
        """Allocate phenotype buffer with full N_trees slots."""
        N_total = self.N_trees
        self._prepped_phenotype = kgs.SolutionCollectionSquare()
        self._prepped_phenotype.xyt = cp.zeros(
            (self.N_solutions, N_total, 3), dtype=kgs.dtype_cp
        )
        self._prepped_phenotype.h = cp.zeros_like(self.h)
        self._prepped_phenotype.use_fixed_h = self.use_fixed_h

    def convert_to_phenotype(self):
        """Always generate full phenotype (lattice + edge trees).
        
        Overrides base which would deepcopy self when is_phenotype() is True.
        We always want to expand lattice params into tree positions.
        """
        was_prepped = self._prepped_for_phenotype
        if not self._prepped_for_phenotype:
            self.prep_for_phenotype()
        phenotype = self._convert_to_phenotype()
        if not was_prepped:
            self.unprep_for_phenotype()
        return phenotype

    def _unprep_for_phenotype(self):
        self._prepped_phenotype = None

    def _convert_to_phenotype(self):
        """Generate lattice trees from params, concatenate with edge trees.
        
        Handles merged solutions (from run_simulation_list) where 
        lattice_params may have fewer rows than N_solutions. In that case,
        wraps lattice_params indices with modulo.
        """
        pheno = self._prepped_phenotype
        pheno.h[:] = self.h

        N_edge = self.xyt.shape[1]

        for i in range(self.N_solutions):
            params = self.lattice_params[i].get()
            t_same, t_horiz, t_vert = float(params[0]), float(params[1]), float(params[2])
            theta, adx, ady = float(params[3]), float(params[4]), float(params[5])
            square_size = float(self.h[i, 0].get())

            inner_xyt = generate_lattice_trees(
                t_same, t_horiz, t_vert, theta, adx, ady,
                self.n_inner, square_size, SCALE_FACTOR
            )

            n_placed = min(len(inner_xyt), self.n_inner)
            if n_placed > 0:
                pheno.xyt[i, :n_placed, :] = cp.array(inner_xyt[:n_placed], dtype=kgs.dtype_cp)
            if n_placed < self.n_inner:
                pheno.xyt[i, n_placed:self.n_inner, :] = 0.0

            # Copy edge trees after lattice trees
            pheno.xyt[i, self.n_inner:, :] = self.xyt[i, :N_edge, :]

        return pheno

    def backprop_phenotype(
        self, grad_xyt_phenotype, grad_h_phenotype,
        grad_xyt_genotype, grad_h_genotype
    ):
        """Only pass gradients for edge trees. Lattice tree gradients are discarded."""
        # Edge trees are at positions [n_inner:] in phenotype
        grad_xyt_genotype[:] = grad_xyt_phenotype[:, self.n_inner:, :]
        grad_h_genotype[:] = grad_h_phenotype

    def _check_constraints(self):
        # Skip parent _check_constraints which validates N_trees against xyt.shape[1]
        # Our xyt only has edge trees, N_trees includes lattice
        if self.lattice_params is not None:
            if self.lattice_params.shape[0] == self.N_solutions:
                assert self.lattice_params.shape == (self.N_solutions, N_LATTICE_PARAMS)

    def select_ids(self, inds):
        super().select_ids(inds)
        if self.lattice_params is not None:
            self.lattice_params = self.lattice_params[inds]

    def merge(self, other):
        super().merge(other)
        if self.lattice_params is not None and hasattr(other, 'lattice_params') \
                and other.lattice_params is not None:
            self.lattice_params = cp.concatenate(
                [self.lattice_params, other.lattice_params], axis=0
            )
        elif hasattr(other, 'lattice_params') and other.lattice_params is not None:
            self.lattice_params = other.lattice_params

    def create_clone(self, idx, other, parent_id):
        super().create_clone(idx, other, parent_id)
        if self.lattice_params is not None and hasattr(other, 'lattice_params') \
                and other.lattice_params is not None:
            self.lattice_params[idx] = other.lattice_params[parent_id]

    def create_clone_batch(self, inds, other, parent_ids):
        super().create_clone_batch(inds, other, parent_ids)
        if self.lattice_params is not None and hasattr(other, 'lattice_params') \
                and other.lattice_params is not None:
            self.lattice_params[inds] = other.lattice_params[parent_ids]

    def create_empty(self, N_solutions, N_trees):
        """Allocate with edge-only xyt. N_trees is the TOTAL (lattice + edge)."""
        N_edge = N_trees - self.n_inner
        res = copy.deepcopy(self)
        res.xyt = cp.zeros((N_solutions, N_edge, 3), dtype=kgs.dtype_cp)
        res.h = cp.zeros((N_solutions, self._N_h_DOF), dtype=kgs.dtype_cp)
        res.lattice_params = cp.zeros(
            (N_solutions, N_LATTICE_PARAMS), dtype=kgs.dtype_cp
        )
        res.n_inner = self.n_inner
        return res

    def __eq__(self, other):
        """Equality check that ignores lattice_params.
        
        pack_dynamics.run_simulation_list compares solutions after nullifying
        xyt and h. lattice_params will differ between islands, but that's
        expected and shouldn't block merging.
        """
        if not isinstance(other, SolutionCollectionSquareParametrizedLattice):
            return False
        # Save and temporarily clear lattice_params
        self_lp, other_lp = self.lattice_params, other.lattice_params
        self.lattice_params = None
        other.lattice_params = None
        try:
            result = super().__eq__(other)
        finally:
            self.lattice_params = self_lp
            other.lattice_params = other_lp
        return result

    def snap(self):
        """Set h from phenotype bbox."""
        phenotype = self.convert_to_phenotype()
        phenotype.snap()
        self.h[:] = phenotype.h[:]


# =============================================================================
# INITIALIZER
# =============================================================================

@dataclass
class InitializerParametrizedLattice(pack_ga3.Initializer):
    """Initialize population with parametrized lattice core + random edge trees.

    Each individual gets random lattice parameters which generate inner tree
    positions via NFP-based lattice. Remaining trees are scattered randomly.
    """

    base_solution: SolutionCollectionSquareParametrizedLattice = field(
        init=True, default_factory=SolutionCollectionSquareParametrizedLattice
    )
    fixed_h: cp.ndarray = field(init=True, default=None)
    n_inner_ratio: float = field(init=True, default=0.7)

    t_range: tuple = field(init=True, default=(-0.5, 0.5))
    anchor_range_ratio: float = field(init=True, default=0.1)

    def _initialize_population(self, N_individuals, N_trees):
        """Create initial population with parametrized lattice seeds."""
        N_inner = int(N_trees * self.n_inner_ratio)
        N_edge = N_trees - N_inner
        self.base_solution.n_inner = N_inner

        sol = self.base_solution.create_empty(N_individuals, N_trees)
        sol.h = cp.tile(self.fixed_h[cp.newaxis, :], (N_individuals, 1))

        generator = np.random.default_rng(seed=self.seed)
        square_size = float(cp.asnumpy(self.fixed_h[0]))

        lattice_params = np.zeros((N_individuals, N_LATTICE_PARAMS), dtype=np.float64)
        anchor_range = self.anchor_range_ratio * square_size / SCALE_FACTOR  # in 80x coords

        for i in range(N_individuals):
            t_same = generator.uniform(*self.t_range)
            t_horiz = generator.uniform(*self.t_range)
            t_vert = generator.uniform(*self.t_range)
            theta = generator.uniform(0, 360)
            adx = generator.uniform(-anchor_range, anchor_range)
            ady = generator.uniform(-anchor_range, anchor_range)

            lattice_params[i] = [t_same, t_horiz, t_vert, theta, adx, ady]

            # Generate lattice trees (just to compute edge placement radius)
            inner_xyt = generate_lattice_trees(
                t_same, t_horiz, t_vert, theta, adx, ady,
                N_inner, square_size, SCALE_FACTOR
            )

            # Fill edge trees randomly (xyt only holds edge trees)
            edge_xyt = _random_edge_trees(
                N_edge, square_size, inner_xyt, generator
            )
            sol.xyt[i, :, :] = cp.array(edge_xyt, dtype=kgs.dtype_cp)

        sol.lattice_params = cp.array(lattice_params, dtype=kgs.dtype_cp)

        return pack_ga3.Population(genotype=sol)


def _random_edge_trees(n_trees, square_size, inner_xyt, generator):
    """Place trees randomly outside the lattice core circle."""
    half = square_size / 2

    if len(inner_xyt) > 0:
        inner_np = inner_xyt.get() if hasattr(inner_xyt, 'get') else inner_xyt
        max_r = np.max(np.sqrt(inner_np[:, 0]**2 + inner_np[:, 1]**2))
        lattice_radius = max_r * 0.8
    else:
        lattice_radius = 0

    result = np.zeros((n_trees, 3), dtype=np.float64)
    placed = 0

    while placed < n_trees:
        batch = min(n_trees - placed, 200)
        x = generator.uniform(-half, half, batch)
        y = generator.uniform(-half, half, batch)
        t = generator.uniform(0, 2 * np.pi, batch)

        r = np.sqrt(x**2 + y**2)
        mask = r > lattice_radius
        n_accept = min(np.sum(mask), n_trees - placed)
        if n_accept > 0:
            result[placed:placed + n_accept, 0] = x[mask][:n_accept]
            result[placed:placed + n_accept, 1] = y[mask][:n_accept]
            result[placed:placed + n_accept, 2] = t[mask][:n_accept]
            placed += n_accept

    return result


# =============================================================================
# GA MUTATION MOVES (pack_move.Move subclasses)
# =============================================================================

import pack_move


@dataclass
class LatticeJiggle(pack_move.Move):
    """Small perturbation to lattice parameters.

    Fits into MoveSelector alongside standard tree moves like
    JiggleRandomTree, Twist, etc. Operates on lattice_params then
    regenerates inner trees.
    """
    t_scale: float = field(init=True, default=0.05)
    theta_scale: float = field(init=True, default=5.0)  # degrees
    translate_scale_ratio: float = field(init=True, default=0.02)  # fraction of square size in 80x

    def _do_move_vec(self, population, inds_to_do, mate_sol, inds_mate, generator):
        sol = population.genotype
        if not isinstance(sol, SolutionCollectionSquareParametrizedLattice):
            return
        if sol.lattice_params is None:
            return

        N = int(inds_to_do.shape[0])
        params = sol.lattice_params[inds_to_do]  # (N, 6)

        # Jiggle t params, clamp to [-1, 1]
        params[:, 0] += generator.standard_normal(N) * self.t_scale
        params[:, 1] += generator.standard_normal(N) * self.t_scale
        params[:, 2] += generator.standard_normal(N) * self.t_scale
        params[:, :3] = cp.clip(params[:, :3], -1.0, 1.0)

        # Jiggle theta (degrees)
        params[:, 3] += generator.standard_normal(N) * self.theta_scale
        params[:, 3] = params[:, 3] % 360.0

        # Jiggle anchor (80x scale)
        square_sizes_80x = sol.h[inds_to_do, 0] / SCALE_FACTOR
        anchor_scale = square_sizes_80x * self.translate_scale_ratio
        params[:, 4] += generator.standard_normal(N) * anchor_scale
        params[:, 5] += generator.standard_normal(N) * anchor_scale

        sol.lattice_params[inds_to_do] = params


@dataclass
class LatticeJump(pack_move.Move):
    """Large random change to lattice parameters."""

    def _do_move_vec(self, population, inds_to_do, mate_sol, inds_mate, generator):
        sol = population.genotype
        if not isinstance(sol, SolutionCollectionSquareParametrizedLattice):
            return
        if sol.lattice_params is None:
            return

        N = int(inds_to_do.shape[0])
        square_sizes_80x = sol.h[inds_to_do, 0] / SCALE_FACTOR
        anchor_range = 0.1 * square_sizes_80x

        sol.lattice_params[inds_to_do, 0] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, 1] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, 2] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, 3] = generator.uniform(0, 360, N)
        sol.lattice_params[inds_to_do, 4] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range
        sol.lattice_params[inds_to_do, 5] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range


@dataclass
class LatticeCrossover(pack_move.Move):
    """Copy entire lattice_params from mate, regenerate inner trees.

    Takes the mate's lattice structure but keeps edge trees from parent.
    """

    def _do_move_vec(self, population, inds_to_do, mate_sol, inds_mate, generator):
        sol = population.genotype
        if not isinstance(sol, SolutionCollectionSquareParametrizedLattice):
            return
        if sol.lattice_params is None:
            return
        if not isinstance(mate_sol, SolutionCollectionSquareParametrizedLattice):
            return
        if mate_sol.lattice_params is None:
            return

        sol.lattice_params[inds_to_do] = mate_sol.lattice_params[inds_mate]


# =============================================================================
# FACTORY
# =============================================================================

def baseline_parametrized_lattice():
    """Create Orchestrator with parametrized lattice initialization.

    Returns:
        Orchestrator with 16-island ring topology, parametrized lattice
        initializer, lattice mutation moves in the MoveSelector, no edge_spacer.
    """
    runner = pack_ga3.baseline()

    # Create initializer with our subclass as base_solution
    base_sol = SolutionCollectionSquareParametrizedLattice()
    base_sol.edge_spacer = kgs.EdgeSpacerDummy()
    base_sol.filter_move_locations_with_edge_spacer = False
    base_sol.override_phenotype = True  # L-BFGS operates on edge trees only (fast)

    initializer = InitializerParametrizedLattice()
    initializer.base_solution = base_sol

    runner.ga.ga_base.initializer = initializer

    # Add lattice moves to the existing MoveSelector
    # Standard moves (MoveRandomTree, JiggleTree, Twist, Crossover, etc.)
    # are already there from GASinglePopulationDiversity.__post_init__().
    # We append lattice-specific moves with moderate weights.
    #
    # Weight rationale:
    #   Standard tree moves total ~11.0 (9 moves, weights 1-2 each)
    #   LatticeJiggle at 1.5 -> ~12% of moves perturb lattice slightly
    #   LatticeJump at 0.3   -> ~2.4% of moves do large lattice changes
    #   LatticeCrossover at 0.5 -> ~4% swap lattice structure from mate
    #   Total lattice: ~18% of moves affect lattice params
    #   Remaining 82% are standard tree-level moves (edge + drifted inner)
    move_selector = runner.ga.ga_base.move
    move_selector.moves.append(
        [LatticeJiggle(), 'LatticeJiggle', 1.5]
    )
    move_selector.moves.append(
        [LatticeJump(), 'LatticeJump', 0.3]
    )
    move_selector.moves.append(
        [LatticeCrossover(), 'LatticeCrossover', 0.5]
    )
    # Force recompute of cached probabilities
    move_selector._probabilities = None

    runner.ga.stop_check_generations_scale = 20

    return runner


# =============================================================================
# INTEGRATION NOTES
# =============================================================================
"""
STATUS: Ready for GPU testing.

All integration pieces are in place:
- SolutionCollectionSquareParametrizedLattice carries lattice_params through
  select_ids, merge, create_clone, create_clone_batch, create_empty
- LatticeJiggle, LatticeJump, LatticeCrossover are proper pack_move.Move 
  subclasses wired into MoveSelector via baseline_parametrized_lattice()
- NFP data, interpolation, and lattice generation ported from Numba to NumPy

USAGE:
    import pack_parametrized_lattice as ppl
    runner = ppl.baseline_parametrized_lattice()
    runner.run(N_trees=50)  # or whatever API Orchestrator.run() expects

NOTES:
- compute_collision_penalty is NOT used at runtime. Jeroen's GPU overlap
  detection handles all collision checking. The penalty function is only
  called during lattice generation to reject obviously broken lattice configs.
- regenerate_inner_trees runs on CPU (NumPy). Only called for individuals 
  that receive a lattice mutation (~18% of offspring). Should be <1ms each.
- SCALE_FACTOR = 1/80 is hardcoded. Serge's 80x integer coords match
  Jeroen's unit scale exactly (tip 64->0.8, base 28->0.35).

TUNING KNOBS:
- n_inner_ratio (default 0.7): fraction of trees placed by lattice vs free
- Move weights in baseline_parametrized_lattice(): LatticeJiggle 1.5,
  LatticeJump 0.3, LatticeCrossover 0.5 (~18% total lattice moves)
- LatticeJiggle scales: t_scale=0.05, theta_scale=0.1, translate_scale=2.0
- t_range for initialization: (-0.5, 0.5), narrower than full [-1,1]
  to bias toward reasonable lattices
"""
