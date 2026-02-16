"""
Parametrized lattice integration for tree packing GA.

Ports Serge's NFP-based lattice parametrization from Numba to CuPy/NumPy
for integration with Jeroen Cottaar's GPU-accelerated GA framework.

The lattice is defined by 8 continuous parameters:
  t_same, t_horiz, t_vert: NFP walk parameters [-1, 1]
  theta:  whole-lattice rotation angle [0, 360] degrees
  dx, dy: lattice anchor translation from square center (80x scale)
  n_inner: number of lattice-controlled trees (integer stored as float)
  p: selection shape parameter (1=square, sqrt(2)=circle)

xyt always holds ALL N_total trees (always-phenotype design):
  xyt[:, :n_inner, :] = lattice-generated trees (frozen during L-BFGS)
  xyt[:, n_inner:, :] = free edge trees (optimized by L-BFGS)

Lattice mutations update params and regenerate xyt[:, :n_inner].
n_inner can vary per solution; crossover picks one parent's n_inner.

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
                           n_inner, square_size, scale=1.0, p=2.0,
                           shift_dx=0.0, shift_dy=0.0):
    """Generate inner tree positions from parametrized lattice.

    Always returns exactly n_inner trees, sorted by selection metric.

    Args:
        t_same, t_horiz, t_vert: NFP walk parameters [-1, 1]
        theta: lattice rotation angle in DEGREES
        anchor_dx, anchor_dy: lattice anchor offset (80x scale), affects selection
        n_inner: exact number of inner trees to return
        square_size: current square boundary size (in Jeroen's scale)
        scale: coordinate scale factor (1/80)
        p: selection shape (1=square, >=sqrt(2)=circle)
        shift_dx, shift_dy: post-selection shift (80x scale), positions cluster in square

    Returns:
        xyt: NumPy array (n_inner, 3) in Jeroen's coordinate system
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

    # 4. Convert lattice positions to tree origins and apply anchor
    orient = positions[:, 2]
    adx = float(anchor_dx)
    ady = float(anchor_dy)
    tree_x = positions[:, 0] + bbx * (2 * orient - 1) + adx
    tree_y = positions[:, 1] + bby * (2 * orient - 1) + ady

    # 5. Sort by selection metric: max(p*dx^2, p*dy^2, dx^2+dy^2)
    # p=1: square selection, p>=sqrt(2): circle selection
    p_val = float(p)
    dx2 = tree_x ** 2
    dy2 = tree_y ** 2
    dist = np.maximum(p_val * dx2, np.maximum(p_val * dy2, dx2 + dy2))
    order = np.argsort(dist)
    n_select = min(n_inner, len(order))
    sel = order[:n_select]

    # 6. Build result with post-selection shift
    sdx = float(shift_dx)
    sdy = float(shift_dy)
    result = np.empty((n_select, 3), dtype=np.float64)
    result[:, 0] = (tree_x[sel] + sdx) * scale
    result[:, 1] = (tree_y[sel] + sdy) * scale
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

N_LATTICE_PARAMS = 10

# Parameter indices
LP_T_SAME = 0
LP_T_HORIZ = 1
LP_T_VERT = 2
LP_THETA = 3
LP_ANCHOR_DX = 4
LP_ANCHOR_DY = 5
LP_N_INNER = 6
LP_P = 7
LP_SHIFT_DX = 8
LP_SHIFT_DY = 9


@dataclass
class SolutionCollectionSquareParametrizedLattice(kgs.SolutionCollectionSquare):
    """Square boundary with parametrized lattice core (always-phenotype).

    xyt always holds ALL N_total trees:
      xyt[:, :n_inner_i, :] = lattice-generated (frozen during L-BFGS)
      xyt[:, n_inner_i:, :] = free edge trees (optimized by L-BFGS)
    
    n_inner is per-solution, stored in lattice_params[:, LP_N_INNER].
    
    Fields:
      lattice_params: (N_solutions, 8) lattice parameters
    """
    lattice_params: cp.ndarray = field(init=True, default=None)

    @property
    def n_inner_array(self):
        """Per-solution n_inner as numpy int array."""
        if self.lattice_params is None:
            return np.zeros(self.N_solutions, dtype=int)
        return cp.asnumpy(self.lattice_params[:, LP_N_INNER]).astype(int)

    @property
    def n_inner_max(self):
        """Maximum n_inner across all solutions (for frozen prefix)."""
        arr = self.n_inner_array
        return int(arr.max()) if len(arr) > 0 else 0

    def is_phenotype(self):
        """Always phenotype — xyt holds all trees."""
        return True

    def regenerate_lattice_trees(self, inds=None):
        """Regenerate lattice portion of xyt from params.
        
        Args:
            inds: solution indices to regenerate (None = all)
        """
        if self.lattice_params is None:
            return
        if inds is None:
            inds = range(self.N_solutions)

        n_inner_arr = self.n_inner_array

        for i in inds:
            i_int = int(i)
            n_inner_i = n_inner_arr[i_int]
            if n_inner_i == 0:
                continue

            params = self.lattice_params[i_int].get()
            t_same = float(params[LP_T_SAME])
            t_horiz = float(params[LP_T_HORIZ])
            t_vert = float(params[LP_T_VERT])
            theta = float(params[LP_THETA])
            adx = float(params[LP_ANCHOR_DX])
            ady = float(params[LP_ANCHOR_DY])
            p = float(params[LP_P])
            sdx = float(params[LP_SHIFT_DX])
            sdy = float(params[LP_SHIFT_DY])
            square_size = float(self.h[i_int, 0].get())

            inner_xyt = generate_lattice_trees(
                t_same, t_horiz, t_vert, theta, adx, ady,
                n_inner_i, square_size, SCALE_FACTOR, p, sdx, sdy
            )

            n_placed = min(len(inner_xyt), n_inner_i)
            if n_placed > 0:
                self.xyt[i_int, :n_placed, :] = cp.array(
                    inner_xyt[:n_placed], dtype=kgs.dtype_cp
                )
            if n_placed < n_inner_i:
                self.xyt[i_int, n_placed:n_inner_i, :] = 0.0

    def get_n_frozen(self):
        """Get max frozen prefix for L-BFGS gradient masking."""
        return self.n_inner_max

    def _check_constraints(self):
        if self.lattice_params is not None:
            if self.lattice_params.shape[0] == self.N_solutions:
                assert self.lattice_params.shape[1] == N_LATTICE_PARAMS

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
        """Allocate with full N_trees xyt (always-phenotype)."""
        res = copy.deepcopy(self)
        res.xyt = cp.zeros((N_solutions, N_trees, 3), dtype=kgs.dtype_cp)
        res.h = cp.zeros((N_solutions, self._N_h_DOF), dtype=kgs.dtype_cp)
        res.lattice_params = cp.zeros(
            (N_solutions, N_LATTICE_PARAMS), dtype=kgs.dtype_cp
        )
        return res

    def __eq__(self, other):
        """Equality check that ignores lattice_params."""
        if not isinstance(other, SolutionCollectionSquareParametrizedLattice):
            return False
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
        """Set h from bbox of all trees."""
        super().snap()


# =============================================================================
# INITIALIZER
# =============================================================================

@dataclass
class InitializerParametrizedLattice(pack_ga3.Initializer):
    """Initialize population with parametrized lattice core + random edge trees.

    Each individual gets random lattice parameters which generate inner tree
    positions via NFP-based lattice. Remaining trees are scattered randomly.
    n_inner and p are per-solution, stored in lattice_params.
    """

    base_solution: SolutionCollectionSquareParametrizedLattice = field(
        init=True, default_factory=SolutionCollectionSquareParametrizedLattice
    )
    fixed_h: cp.ndarray = field(init=True, default=None)
    n_inner_ratio: float = field(init=True, default=0.7)
    p_range: tuple = field(init=True, default=(1.0, 1.5))

    t_range: tuple = field(init=True, default=(-0.5, 0.5))
    anchor_range_ratio: float = field(init=True, default=0.1)

    def _initialize_population(self, N_individuals, N_trees):
        """Create initial population with parametrized lattice seeds."""
        N_inner = int(N_trees * self.n_inner_ratio)

        sol = self.base_solution.create_empty(N_individuals, N_trees)
        sol.h = cp.tile(self.fixed_h[cp.newaxis, :], (N_individuals, 1))

        generator = np.random.default_rng(seed=self.seed)
        square_size = float(cp.asnumpy(self.fixed_h[0]))

        lattice_params = np.zeros((N_individuals, N_LATTICE_PARAMS), dtype=np.float64)
        anchor_range = self.anchor_range_ratio * square_size / SCALE_FACTOR

        for i in range(N_individuals):
            t_same = generator.uniform(*self.t_range)
            t_horiz = generator.uniform(*self.t_range)
            t_vert = generator.uniform(*self.t_range)
            theta = generator.uniform(0, 360)
            adx = generator.uniform(-anchor_range, anchor_range)
            ady = generator.uniform(-anchor_range, anchor_range)
            p = generator.uniform(*self.p_range)

            lattice_params[i] = [t_same, t_horiz, t_vert, theta, adx, ady, N_inner, p, 0.0, 0.0]

            # Generate lattice trees
            inner_xyt = generate_lattice_trees(
                t_same, t_horiz, t_vert, theta, adx, ady,
                N_inner, square_size, SCALE_FACTOR, p
            )

            # Place lattice trees in first n_inner slots
            n_placed = min(len(inner_xyt), N_inner)
            if n_placed > 0:
                sol.xyt[i, :n_placed, :] = cp.array(inner_xyt[:n_placed], dtype=kgs.dtype_cp)

            # Fill edge trees randomly in remaining slots
            N_edge = N_trees - N_inner
            edge_xyt = _random_edge_trees(
                N_edge, square_size, inner_xyt, generator
            )
            sol.xyt[i, N_inner:, :] = cp.array(edge_xyt, dtype=kgs.dtype_cp)

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

    Jiggles t_same/t_horiz/t_vert, theta, anchor, and p.
    Does NOT change n_inner (use LatticeResizeInner for that).
    After jiggling, regenerates lattice trees in xyt.
    """
    t_scale: float = field(init=True, default=0.05)
    theta_scale: float = field(init=True, default=5.0)  # degrees
    translate_scale_ratio: float = field(init=True, default=0.02)
    p_scale: float = field(init=True, default=0.05)

    def _do_move_vec(self, population, inds_to_do, mate_sol, inds_mate, generator):
        sol = population.genotype
        if not isinstance(sol, SolutionCollectionSquareParametrizedLattice):
            return
        if sol.lattice_params is None:
            return

        N = int(inds_to_do.shape[0])
        params = sol.lattice_params[inds_to_do]  # (N, 8)

        # Jiggle t params, clamp to [-1, 1]
        params[:, LP_T_SAME] += generator.standard_normal(N) * self.t_scale
        params[:, LP_T_HORIZ] += generator.standard_normal(N) * self.t_scale
        params[:, LP_T_VERT] += generator.standard_normal(N) * self.t_scale
        params[:, LP_T_SAME:LP_T_VERT+1] = cp.clip(
            params[:, LP_T_SAME:LP_T_VERT+1], -1.0, 1.0
        )

        # Jiggle theta (degrees)
        params[:, LP_THETA] += generator.standard_normal(N) * self.theta_scale
        params[:, LP_THETA] = params[:, LP_THETA] % 360.0

        # Jiggle anchor (80x scale)
        square_sizes_80x = sol.h[inds_to_do, 0] / SCALE_FACTOR
        anchor_scale = square_sizes_80x * self.translate_scale_ratio
        params[:, LP_ANCHOR_DX] += generator.standard_normal(N) * anchor_scale
        params[:, LP_ANCHOR_DY] += generator.standard_normal(N) * anchor_scale

        # Jiggle p, clamp to [1, sqrt(2)]
        params[:, LP_P] += generator.standard_normal(N) * self.p_scale
        params[:, LP_P] = cp.clip(params[:, LP_P], 1.0, 1.42)

        # Jiggle shift (80x scale, same scale as anchor)
        params[:, LP_SHIFT_DX] += generator.standard_normal(N) * anchor_scale
        params[:, LP_SHIFT_DY] += generator.standard_normal(N) * anchor_scale

        sol.lattice_params[inds_to_do] = params

        # Regenerate lattice trees
        sol.regenerate_lattice_trees(cp.asnumpy(inds_to_do))


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

        sol.lattice_params[inds_to_do, LP_T_SAME] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, LP_T_HORIZ] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, LP_T_VERT] = generator.uniform(-0.5, 0.5, N)
        sol.lattice_params[inds_to_do, LP_THETA] = generator.uniform(0, 360, N)
        sol.lattice_params[inds_to_do, LP_ANCHOR_DX] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range
        sol.lattice_params[inds_to_do, LP_ANCHOR_DY] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range
        sol.lattice_params[inds_to_do, LP_P] = generator.uniform(1.0, 1.42, N)
        sol.lattice_params[inds_to_do, LP_SHIFT_DX] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range
        sol.lattice_params[inds_to_do, LP_SHIFT_DY] = (generator.uniform(0, 1, N) * 2 - 1) * anchor_range
        # n_inner unchanged

        sol.regenerate_lattice_trees(cp.asnumpy(inds_to_do))


@dataclass
class LatticeCrossover(pack_move.Move):
    """Copy entire lattice_params from mate, regenerate lattice trees.

    Takes the mate's lattice structure (including n_inner).
    When n_inner differs, the boundary between lattice and free trees moves:
    - If mate has more lattice trees: some free trees become lattice
    - If mate has fewer: some lattice trees become free (keep positions)
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
        sol.regenerate_lattice_trees(cp.asnumpy(inds_to_do))


@dataclass
class LatticeResizeInner(pack_move.Move):
    """Increment or decrement n_inner by 1.
    
    Expanding lattice (n_inner += 1): tree at position n_inner becomes
    lattice-controlled, overwritten by regeneration.
    
    Shrinking lattice (n_inner -= 1): tree at position n_inner-1 becomes
    free, keeps its current position.
    """

    def _do_move_vec(self, population, inds_to_do, mate_sol, inds_mate, generator):
        sol = population.genotype
        if not isinstance(sol, SolutionCollectionSquareParametrizedLattice):
            return
        if sol.lattice_params is None:
            return

        N = int(inds_to_do.shape[0])
        N_trees = sol.xyt.shape[1]

        # Random +1 or -1 for each individual
        delta = cp.where(generator.uniform(0, 1, N) < 0.5, -1.0, 1.0)
        new_n_inner = sol.lattice_params[inds_to_do, LP_N_INNER] + delta

        # Clamp to [1, N_trees - 1] (at least 1 lattice, 1 free)
        new_n_inner = cp.clip(new_n_inner, 1.0, float(N_trees - 1))

        sol.lattice_params[inds_to_do, LP_N_INNER] = new_n_inner

        # Regenerate lattice trees for changed individuals
        sol.regenerate_lattice_trees(cp.asnumpy(inds_to_do))


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

    # Create base solution (always-phenotype, no genotype/phenotype split)
    base_sol = SolutionCollectionSquareParametrizedLattice()
    base_sol.edge_spacer = kgs.EdgeSpacerDummy()
    base_sol.filter_move_locations_with_edge_spacer = False

    initializer = InitializerParametrizedLattice()
    initializer.base_solution = base_sol

    runner.ga.ga_base.initializer = initializer

    # Add lattice moves to the existing MoveSelector
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
    move_selector.moves.append(
        [LatticeResizeInner(), 'LatticeResizeInner', 0.3]
    )
    # Force recompute of cached probabilities
    move_selector._probabilities = None

    runner.ga.stop_check_generations_scale = 20

    return runner


# =============================================================================
# INTEGRATION NOTES
# =============================================================================
"""
STATUS: Always-phenotype design with per-solution n_inner and selection shape p.

xyt always holds ALL N_total trees:
  xyt[:, :n_inner, :] = lattice-generated (frozen during L-BFGS)
  xyt[:, n_inner:, :] = free edge trees (optimized by L-BFGS)

LATTICE PARAMS (8 per solution):
  [t_same, t_horiz, t_vert, theta, anchor_dx, anchor_dy, n_inner, p]
  - n_inner: integer (stored as float), varies per solution
  - p: selection shape (1=square, sqrt(2)=circle), smooth interpolation

MUTATIONS:
  LatticeJiggle (1.5): small perturbation to t/theta/anchor/p
  LatticeJump (0.3): large random reset of lattice params
  LatticeCrossover (0.5): copy entire params from mate
  LatticeResizeInner (0.3): ±1 to n_inner boundary

USAGE:
    import pack_parametrized_lattice as ppl
    runner = ppl.baseline_parametrized_lattice()
    runner.run(N_trees=50)
"""
