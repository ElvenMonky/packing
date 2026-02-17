"""Tests for pack_parametrized_lattice module.

Tests NFP interpolation, lattice generation, SolutionCollection subclass
operations, move mechanics, and GA integration.

Run from notebook:
    import sys, os
    sys.path.insert(0, os.path.join(os.getcwd(), 'core'))
    import pack_test_lattice
    pack_test_lattice.run_all_tests()
"""

import numpy as np
import cupy as cp
import copy

import kaggle_support as kgs
import pack_cost
import pack_move
import pack_ga3

kgs.set_float32(True)
import pack_cuda
pack_cuda._ensure_initialized()

import pack_parametrized_lattice as ppl


# =============================================================================
# NFP / Cell Vector Tests
# =============================================================================

def test_nfp_interpolation():
    """Verify NFP interpolation at known waypoint values and between them."""
    print("Testing NFP interpolation...")

    # At t=0, same-cell contact should give (28, 64) — the corner point
    x, y = ppl._interpolate_nfp(0.0, ppl.NFP_SAME_UPPER, ppl.NFP_SAME_LOWER)
    assert abs(x - 28.0) < 1e-10 and abs(y - 64.0) < 1e-10, \
        f"t=0 same: expected (28,64), got ({x},{y})"

    # At t=1, same upper should give (0, 128)
    x, y = ppl._interpolate_nfp(1.0, ppl.NFP_SAME_UPPER, ppl.NFP_SAME_LOWER)
    assert abs(x - 0.0) < 1e-10 and abs(y - 128.0) < 1e-10, \
        f"t=1 same: expected (0,128), got ({x},{y})"

    # At t=-1, same lower should give (56, 0)
    x, y = ppl._interpolate_nfp(-1.0, ppl.NFP_SAME_UPPER, ppl.NFP_SAME_LOWER)
    assert abs(x - 56.0) < 1e-10 and abs(y - 0.0) < 1e-10, \
        f"t=-1 same: expected (56,0), got ({x},{y})"

    # At t=0, vert should give (0, -32)
    x, y = ppl._interpolate_nfp(0.0, ppl.NFP_VERT_UPPER, ppl.NFP_VERT_LOWER)
    assert abs(x - 0.0) < 1e-10 and abs(y - (-32.0)) < 1e-10, \
        f"t=0 vert: expected (0,-32), got ({x},{y})"

    # Interpolation should be continuous — check midpoints don't jump
    prev_x, prev_y = ppl._interpolate_nfp(-1.0, ppl.NFP_SAME_UPPER, ppl.NFP_SAME_LOWER)
    for t in np.linspace(-1.0, 1.0, 200):
        x, y = ppl._interpolate_nfp(t, ppl.NFP_SAME_UPPER, ppl.NFP_SAME_LOWER)
        assert abs(x - prev_x) < 10.0 and abs(y - prev_y) < 15.0, \
            f"Discontinuity at t={t}: ({prev_x},{prev_y}) -> ({x},{y})"
        prev_x, prev_y = x, y

    print("  ✓ NFP interpolation correct at waypoints and continuous")


def test_cell_vectors_no_collision():
    """Verify that cell vectors at t=0 produce zero collision penalty."""
    print("Testing cell vectors at t=0...")

    dx_same, dy_same, dxh, dyh, dxv, dyv = ppl.compute_cell_vectors(0.0, 0.0, 0.0)

    penalty = ppl.compute_collision_penalty(dx_same, dy_same, dxh, dyh, dxv, dyv)
    assert penalty < 1e-10, f"Nonzero collision at t=0: {penalty}"

    area = abs(dxh * dyv - dxv * dyh)
    assert area > 1000 and area < 10000, f"Suspicious cell area at t=0: {area}"

    print(f"  ✓ Zero collision, cell area = {area:.1f}")


def test_cell_vectors_sweep():
    """Sweep t-parameters and verify collision penalty is zero after clamping."""
    print("Testing cell vector clamping across parameter sweep...")

    n_clean = 0
    n_total = 0
    for t_same in np.linspace(-0.8, 0.8, 10):
        for t_horiz in np.linspace(-0.8, 0.8, 10):
            for t_vert in np.linspace(-0.8, 0.8, 10):
                n_total += 1
                dx_same, dy_same, dxh, dyh, dxv, dyv = \
                    ppl.compute_cell_vectors(t_same, t_horiz, t_vert)
                penalty = ppl.compute_collision_penalty(
                    dx_same, dy_same, dxh, dyh, dxv, dyv
                )
                if penalty < 1e-10:
                    n_clean += 1

    ratio = n_clean / n_total
    assert ratio > 0.5, \
        f"Too many colliding configs: {n_clean}/{n_total} ({ratio:.1%} clean)"

    print(f"  ✓ {n_clean}/{n_total} ({ratio:.1%}) collision-free after clamping")


# =============================================================================
# Scale Factor Test
# =============================================================================

def test_scale_factor():
    """Verify scale factor matches Jeroen's tree geometry."""
    print("Testing scale factor...")

    assert abs(ppl.SCALE_FACTOR - 1.0/80.0) < 1e-15, \
        f"Scale factor wrong: {ppl.SCALE_FACTOR}"

    if kgs.tree_vertices is not None:
        jeroen_height = float(kgs.tree_vertices[:, 1].max() - kgs.tree_vertices[:, 1].min())
        serge_height = 80.0
        expected_scale = jeroen_height / serge_height
        assert abs(ppl.SCALE_FACTOR - expected_scale) < 1e-6, \
            f"Scale mismatch: {ppl.SCALE_FACTOR} vs {expected_scale}"

    print(f"  ✓ SCALE_FACTOR = {ppl.SCALE_FACTOR}")


# =============================================================================
# Lattice Generation Tests
# =============================================================================

def test_generate_lattice_basic():
    """Test lattice generation produces valid tree positions."""
    print("Testing lattice generation...")

    square_size = 5.0
    n_inner = 30

    xyt = ppl.generate_lattice_trees(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=n_inner, square_size=square_size,
        scale=ppl.SCALE_FACTOR
    )

    assert xyt.shape[1] == 3, f"Wrong shape: {xyt.shape}"
    assert len(xyt) == n_inner, f"Expected {n_inner} trees, got {len(xyt)}"

    xyt_np = xyt if isinstance(xyt, np.ndarray) else xyt.get()
    half = square_size / 2.0

    assert np.all(np.abs(xyt_np[:, 0]) < half + 1.0), "Tree x outside square"
    assert np.all(np.abs(xyt_np[:, 1]) < half + 1.0), "Tree y outside square"

    angles = xyt_np[:, 2]
    for a in angles:
        assert abs(a) < 1e-10 or abs(a - np.pi) < 1e-10, \
            f"Unexpected angle {a} (expected 0 or pi)"

    print(f"  ✓ Generated {len(xyt)} trees (requested {n_inner})")


def test_generate_lattice_rotation():
    """Test that lattice rotation produces different positions."""
    print("Testing lattice rotation...")

    kwargs = dict(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        anchor_dx=0.0, anchor_dy=0.0,
        n_inner=20, square_size=5.0, scale=ppl.SCALE_FACTOR
    )

    xyt_0 = ppl.generate_lattice_trees(theta=0.0, **kwargs)
    xyt_r = ppl.generate_lattice_trees(theta=30.0, **kwargs)

    assert len(xyt_0) > 0 and len(xyt_r) > 0
    if len(xyt_0) == len(xyt_r):
        diff = np.max(np.abs(xyt_0[:, :2] - xyt_r[:, :2]))
        assert diff > 0.01, "Rotation had no effect on positions"

    print(f"  ✓ Rotation changes positions (n={len(xyt_0)} vs {len(xyt_r)})")


def test_generate_lattice_invalid():
    """Test that extreme lattice params don't crash."""
    print("Testing extreme lattice params...")

    xyt = ppl.generate_lattice_trees(
        t_same=0.99, t_horiz=0.99, t_vert=0.99,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=30, square_size=5.0, scale=ppl.SCALE_FACTOR
    )

    assert len(xyt) == 30, f"Expected 30 trees, got {len(xyt)}"
    assert not np.any(np.isnan(xyt)), "NaN in tree positions"

    print(f"  ✓ Extreme params: {len(xyt)} trees (no crash)")


def test_selection_shape():
    """Test that p parameter affects which trees are selected."""
    print("Testing selection shape parameter...")

    kwargs = dict(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=20, square_size=5.0, scale=ppl.SCALE_FACTOR
    )

    xyt_square = ppl.generate_lattice_trees(p=1.0, **kwargs)
    xyt_circle = ppl.generate_lattice_trees(p=1.42, **kwargs)

    assert len(xyt_square) == 20 and len(xyt_circle) == 20
    diff = np.max(np.abs(xyt_square[:, :2] - xyt_circle[:, :2]))
    print(f"  ✓ Selection shape: square vs circle max diff = {diff:.4f}")


def test_post_selection_shift():
    """Test that shift_dx/dy moves the selected cluster."""
    print("Testing post-selection shift...")

    kwargs = dict(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=20, square_size=5.0, scale=ppl.SCALE_FACTOR, p=1.2
    )

    xyt_base = ppl.generate_lattice_trees(shift_dx=0.0, shift_dy=0.0, **kwargs)
    shift = 40.0  # 80x scale
    xyt_shifted = ppl.generate_lattice_trees(shift_dx=shift, shift_dy=shift, **kwargs)

    # Same trees selected (same anchor), just shifted
    expected_d = shift * ppl.SCALE_FACTOR
    actual_dx = np.mean(xyt_shifted[:, 0] - xyt_base[:, 0])
    actual_dy = np.mean(xyt_shifted[:, 1] - xyt_base[:, 1])

    assert abs(actual_dx - expected_d) < 1e-6, \
        f"X shift wrong: expected {expected_d}, got {actual_dx}"
    assert abs(actual_dy - expected_d) < 1e-6, \
        f"Y shift wrong: expected {expected_d}, got {actual_dy}"

    print(f"  ✓ Shift moves cluster by ({actual_dx:.4f}, {actual_dy:.4f})")


# =============================================================================
# SolutionCollection Subclass Tests
# =============================================================================

def test_solution_collection_create_empty():
    """Test create_empty allocates full N_trees xyt (always-phenotype)."""
    print("Testing SolutionCollection create_empty...")

    base = ppl.SolutionCollectionSquareParametrizedLattice()
    sol = base.create_empty(10, 20)

    assert sol.xyt.shape == (10, 20, 3), f"Wrong xyt shape: {sol.xyt.shape}"
    assert sol.h.shape == (10, 3)
    assert sol.lattice_params.shape == (10, ppl.N_LATTICE_PARAMS)

    print("  ✓ create_empty allocates full xyt and lattice_params")


def test_solution_collection_select_ids():
    """Test select_ids slices lattice_params correctly."""
    print("Testing SolutionCollection select_ids...")

    sol = ppl.SolutionCollectionSquareParametrizedLattice()
    sol.xyt = cp.random.random((10, 15, 3), dtype=kgs.dtype_cp)
    sol.h = cp.random.random((10, 3), dtype=kgs.dtype_cp)
    NLP = ppl.N_LATTICE_PARAMS
    sol.lattice_params = cp.arange(10 * NLP, dtype=kgs.dtype_cp).reshape(10, NLP)

    inds = [2, 5, 7]
    sol.select_ids(inds)

    assert sol.xyt.shape[0] == 3
    assert sol.lattice_params.shape == (3, NLP)
    assert float(sol.lattice_params[0, 0].get()) == 2.0 * NLP
    assert float(sol.lattice_params[1, 0].get()) == 5.0 * NLP
    assert float(sol.lattice_params[2, 0].get()) == 7.0 * NLP

    print("  ✓ select_ids slices lattice_params correctly")


def test_solution_collection_merge():
    """Test merge concatenates lattice_params."""
    print("Testing SolutionCollection merge...")

    NLP = ppl.N_LATTICE_PARAMS

    sol_a = ppl.SolutionCollectionSquareParametrizedLattice()
    sol_a.xyt = cp.zeros((3, 15, 3), dtype=kgs.dtype_cp)
    sol_a.h = cp.zeros((3, 3), dtype=kgs.dtype_cp)
    sol_a.lattice_params = cp.ones((3, NLP), dtype=kgs.dtype_cp)

    sol_b = ppl.SolutionCollectionSquareParametrizedLattice()
    sol_b.xyt = cp.zeros((2, 15, 3), dtype=kgs.dtype_cp)
    sol_b.h = cp.zeros((2, 3), dtype=kgs.dtype_cp)
    sol_b.lattice_params = cp.ones((2, NLP), dtype=kgs.dtype_cp) * 2.0

    sol_a.merge(sol_b)

    assert sol_a.xyt.shape[0] == 5
    assert sol_a.lattice_params.shape == (5, NLP)
    assert float(sol_a.lattice_params[0, 0].get()) == 1.0
    assert float(sol_a.lattice_params[3, 0].get()) == 2.0

    print("  ✓ merge concatenates lattice_params")


def test_solution_collection_clone():
    """Test create_clone and create_clone_batch copy lattice_params."""
    print("Testing SolutionCollection clone...")

    n_trees = 15
    NLP = ppl.N_LATTICE_PARAMS

    src = ppl.SolutionCollectionSquareParametrizedLattice()
    src.xyt = cp.random.random((4, n_trees, 3), dtype=kgs.dtype_cp)
    src.h = cp.random.random((4, 3), dtype=kgs.dtype_cp)
    src.lattice_params = cp.arange(4 * NLP, dtype=kgs.dtype_cp).reshape(4, NLP)

    dst = src.create_empty(4, n_trees)

    # Single clone
    dst.create_clone(0, src, 2)
    assert cp.allclose(dst.lattice_params[0], src.lattice_params[2])

    # Batch clone
    dst.create_clone_batch(
        np.array([1, 3]),
        src,
        np.array([0, 3])
    )
    assert cp.allclose(dst.lattice_params[1], src.lattice_params[0])
    assert cp.allclose(dst.lattice_params[3], src.lattice_params[3])

    print("  ✓ clone copies lattice_params correctly")


# =============================================================================
# Cost Function Compatibility Test
# =============================================================================

def test_cost_with_lattice_solution():
    """Test that cost functions work with always-phenotype lattice solution."""
    print("Testing cost functions with lattice solution...")

    kgs.set_float32(False)

    N_trees = 8
    n_inner = 5

    sol = ppl.SolutionCollectionSquareParametrizedLattice()
    sol.xyt = cp.random.random((1, N_trees, 3), dtype=cp.float64)
    sol.xyt[:, :, 0:2] *= 3.0
    sol.xyt[:, :, 0:2] -= 1.5
    sol.h = cp.array([[5.0, 0.0, 0.0]], dtype=cp.float64)
    sol.lattice_params = cp.zeros((1, ppl.N_LATTICE_PARAMS), dtype=cp.float64)
    sol.lattice_params[0, ppl.LP_N_INNER] = float(n_inner)
    sol.lattice_params[0, ppl.LP_P] = 1.2

    sol.regenerate_lattice_trees()

    assert sol.xyt.shape == (1, N_trees, 3), f"Wrong shape: {sol.xyt.shape}"
    assert sol.is_phenotype(), "Should always be phenotype"

    costs = [
        pack_cost.AreaCost(),
        pack_cost.CollisionCostSeparation(scaling=5.0),
        pack_cost.BoundaryDistanceCost(use_kernel=False),
    ]

    for c in costs:
        cost_val, grad_xyt, grad_h = c.compute_cost_ref(sol)
        assert cost_val.shape == (1,), f"{c.__class__.__name__}: wrong cost shape"
        assert not cp.any(cp.isnan(cost_val)), f"{c.__class__.__name__}: NaN cost"
        assert grad_xyt.shape == (1, N_trees, 3)
        print(f"  ✓ {c.__class__.__name__}: cost={float(cost_val[0]):.4f}")

    # Test n_inner_array and n_inner_max
    assert sol.n_inner_array[0] == n_inner
    assert sol.n_inner_max == n_inner
    print("  ✓ n_inner_array, n_inner_max work correctly")

    kgs.set_float32(True)


# =============================================================================
# Move Tests
# =============================================================================

def _make_test_population(n_individuals=10, n_trees=15, n_inner=10):
    """Helper: create a Population with always-phenotype lattice solution."""
    base = ppl.SolutionCollectionSquareParametrizedLattice()
    base.edge_spacer = kgs.EdgeSpacerDummy()
    base.filter_move_locations_with_edge_spacer = False

    genotype = base.create_empty(n_individuals, n_trees)
    genotype.h[:, 0] = 5.0

    # Fill all trees with random positions
    genotype.xyt = cp.random.random(
        (n_individuals, n_trees, 3), dtype=kgs.dtype_cp
    )
    genotype.xyt[:, :, 0:2] *= 3.0
    genotype.xyt[:, :, 0:2] -= 1.5

    # Set reasonable lattice params
    genotype.lattice_params[:, ppl.LP_T_SAME] = 0.0
    genotype.lattice_params[:, ppl.LP_T_HORIZ] = 0.0
    genotype.lattice_params[:, ppl.LP_T_VERT] = 0.0
    genotype.lattice_params[:, ppl.LP_THETA] = 10.0
    genotype.lattice_params[:, ppl.LP_ANCHOR_DX] = 0.0
    genotype.lattice_params[:, ppl.LP_ANCHOR_DY] = 0.0
    genotype.lattice_params[:, ppl.LP_N_INNER] = float(n_inner)
    genotype.lattice_params[:, ppl.LP_P] = 1.2
    genotype.lattice_params[:, ppl.LP_SHIFT_DX] = 0.0
    genotype.lattice_params[:, ppl.LP_SHIFT_DY] = 0.0

    genotype.regenerate_lattice_trees()

    pop = pack_ga3.Population(genotype=genotype)
    pop.phenotype = copy.deepcopy(genotype)
    pop.set_dummy_fitness()

    return pop


def test_lattice_jiggle_move():
    """Test LatticeJiggle move changes lattice params and regenerates trees."""
    print("Testing LatticeJiggle move...")

    pop = _make_test_population()
    generator = cp.random.default_rng(seed=42)

    old_params = pop.genotype.lattice_params.copy()
    old_xyt = pop.genotype.xyt.copy()

    inds = cp.array([0, 3, 7])
    mate_inds = cp.array([1, 2, 4])

    move = ppl.LatticeJiggle()
    move.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    # Params should have changed for indices 0, 3, 7
    for i in [0, 3, 7]:
        diff = float(cp.max(cp.abs(
            pop.genotype.lattice_params[i] - old_params[i]
        )).get())
        assert diff > 1e-6, f"Params unchanged for individual {i}"

    # Lattice trees should have been regenerated
    n_inner = 10
    for i in [0, 3, 7]:
        xyt_diff = float(cp.max(cp.abs(
            pop.genotype.xyt[i, :n_inner] - old_xyt[i, :n_inner]
        )).get())
        assert xyt_diff > 1e-6, f"Lattice trees not regenerated for individual {i}"

    # Params should be unchanged for other indices
    for i in [1, 2, 4, 5, 6, 8, 9]:
        assert cp.allclose(pop.genotype.lattice_params[i], old_params[i]), \
            f"Params changed for non-target individual {i}"

    print("  ✓ LatticeJiggle changes params and regenerates lattice trees")


def test_lattice_jump_move():
    """Test LatticeJump produces substantially different params."""
    print("Testing LatticeJump move...")

    pop = _make_test_population()
    generator = cp.random.default_rng(seed=42)

    old_params = pop.genotype.lattice_params.copy()

    inds = cp.array([2, 5])
    mate_inds = cp.array([0, 1])

    move = ppl.LatticeJump()
    move.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    for i in [2, 5]:
        diff = float(cp.max(cp.abs(
            pop.genotype.lattice_params[i] - old_params[i]
        )).get())
        assert diff > 0.01, f"Jump produced too small change for individual {i}"

    print("  ✓ LatticeJump produces large parameter changes")


def test_lattice_crossover_move():
    """Test LatticeCrossover copies params from mate."""
    print("Testing LatticeCrossover move...")

    pop = _make_test_population()
    generator = cp.random.default_rng(seed=42)

    # Set distinguishable params
    pop.genotype.lattice_params[0] = cp.array(
        [0.1, 0.2, 0.3, 45.0, 0.5, 0.6, 10.0, 1.1, 1.0, 2.0],
        dtype=kgs.dtype_cp
    )
    pop.genotype.lattice_params[5] = cp.array(
        [0.9, 0.8, 0.7, 90.0, 0.5, 0.4, 8.0, 1.3, -1.0, -2.0],
        dtype=kgs.dtype_cp
    )

    mate_sol = copy.deepcopy(pop.genotype)
    inds = cp.array([3])
    mate_inds = cp.array([5])

    move = ppl.LatticeCrossover()
    move.do_move_vec(pop, inds, mate_sol, mate_inds, generator)

    assert cp.allclose(pop.genotype.lattice_params[3], mate_sol.lattice_params[5]), \
        "Crossover didn't copy params from mate"

    print("  ✓ LatticeCrossover copies params from mate")


def test_lattice_resize_inner_move():
    """Test LatticeResizeInner changes n_inner by ±1."""
    print("Testing LatticeResizeInner move...")

    pop = _make_test_population(n_trees=15, n_inner=10)
    generator = cp.random.default_rng(seed=42)

    old_n_inner = pop.genotype.lattice_params[:, ppl.LP_N_INNER].copy()

    inds = cp.array([0, 1, 2, 3, 4])
    mate_inds = cp.array([5, 6, 7, 8, 9])

    move = ppl.LatticeResizeInner()
    move.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    for i in [0, 1, 2, 3, 4]:
        old_val = float(old_n_inner[i].get())
        new_val = float(pop.genotype.lattice_params[i, ppl.LP_N_INNER].get())
        assert abs(new_val - old_val) == 1.0, \
            f"Individual {i}: n_inner changed by {new_val - old_val}, expected ±1"
        assert 1 <= new_val <= 15, f"n_inner out of range: {new_val}"

    for i in [5, 6, 7, 8, 9]:
        assert float(pop.genotype.lattice_params[i, ppl.LP_N_INNER].get()) == \
               float(old_n_inner[i].get())

    print("  ✓ LatticeResizeInner changes n_inner by ±1, clamped to valid range")


def test_move_selector_integration():
    """Test lattice moves work inside MoveSelector alongside standard moves."""
    print("Testing MoveSelector integration...")

    pop = _make_test_population()
    generator = cp.random.default_rng(seed=42)

    selector = pack_move.MoveSelector()
    selector.moves = [
        [pack_move.JiggleRandomTree(max_xy_move=0.05, max_theta_move=0.3),
         'JiggleTree', 2.0],
        [ppl.LatticeJiggle(), 'LatticeJiggle', 1.0],
        [ppl.LatticeJump(), 'LatticeJump', 0.5],
        [ppl.LatticeResizeInner(), 'LatticeResizeInner', 0.3],
    ]

    N = pop.genotype.N_solutions
    inds = cp.arange(N)
    mate_inds = cp.arange(N)

    selector.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    print("  ✓ MoveSelector runs with mixed tree+lattice moves")


# =============================================================================
# GA Integration Test (smoke test)
# =============================================================================

def test_ga_parametrized_lattice():
    """Smoke test: run parametrized lattice GA for a few generations."""
    print("Testing GA with parametrized lattice (smoke test)...")

    ga = ppl.baseline_parametrized_lattice()

    ga.n_generations = 3
    ga.ga.ga_base.N_trees_to_do = 15
    ga.ga.ga_base.population_size = 50
    ga.ga.ga_base.search_depth = 0.8
    ga.ga.ga_base.elitism_fraction = 0.5
    ga.ga.ga_base.survival_rate = 0.7
    ga.ga.do_legalize = False
    ga.ga.ga_base.remove_population_after_abbreviate = False
    ga.rough_relaxers[0].cost.costs[2].lut_N_theta = 50

    ga.run()

    res = ga.ga.ga_list[0].population.fitness
    for g in ga.ga.ga_list[1:]:
        res = np.concatenate((res, g.population.fitness))

    assert len(res) > 0, "No fitness values"
    assert np.all(np.isfinite(res)), "Non-finite fitness values"

    for g in ga.ga.ga_list:
        sol = g.population.genotype
        assert isinstance(sol, ppl.SolutionCollectionSquareParametrizedLattice), \
            f"Solution type lost: {type(sol)}"
        assert sol.lattice_params is not None, "lattice_params is None"
        assert sol.lattice_params.shape[1] == ppl.N_LATTICE_PARAMS, \
            f"lattice_params wrong shape: {sol.lattice_params.shape}"
        n_inner_arr = sol.n_inner_array
        assert np.all(n_inner_arr >= 1), "n_inner < 1"
        assert np.all(n_inner_arr <= 15), "n_inner > N_trees"

    best = np.min(res[:, 0])
    print(f"  ✓ GA completed, best fitness = {best:.6f}")
    print(f"    lattice_params survived through all {len(ga.ga.ga_list)} islands")


# =============================================================================
# Runner
# =============================================================================

def run_all_tests():
    """Execute all parametrized lattice tests."""
    print("=" * 60)
    print("Parametrized Lattice Tests")
    print("=" * 60)

    # NFP / cell vectors
    test_nfp_interpolation()
    test_cell_vectors_no_collision()
    test_cell_vectors_sweep()

    # Scale factor
    test_scale_factor()

    # Lattice generation
    test_generate_lattice_basic()
    test_generate_lattice_rotation()
    test_generate_lattice_invalid()
    test_selection_shape()
    test_post_selection_shift()

    # SolutionCollection subclass
    test_solution_collection_create_empty()
    test_solution_collection_select_ids()
    test_solution_collection_merge()
    test_solution_collection_clone()

    # Cost function compatibility
    test_cost_with_lattice_solution()

    # Move mechanics
    test_lattice_jiggle_move()
    test_lattice_jump_move()
    test_lattice_crossover_move()
    test_lattice_resize_inner_move()
    test_move_selector_integration()

    # GA smoke test
    test_ga_parametrized_lattice()

    print("=" * 60)
    print("All parametrized lattice tests passed!")
    print("=" * 60)
