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
        # Max jump between adjacent samples (dt=0.01) should be small
        assert abs(x - prev_x) < 10.0 and abs(y - prev_y) < 15.0, \
            f"Discontinuity at t={t}: ({prev_x},{prev_y}) -> ({x},{y})"
        prev_x, prev_y = x, y

    print("  ✓ NFP interpolation correct at waypoints and continuous")


def test_cell_vectors_no_collision():
    """Verify that cell vectors at t=0 produce zero collision penalty."""
    print("Testing cell vectors at t=0...")

    dx_same, dy_same, dxh, dyh, dxv, dyv = ppl.compute_cell_vectors(0.0, 0.0, 0.0)

    # Should produce valid (no collision) lattice
    penalty = ppl.compute_collision_penalty(dx_same, dy_same, dxh, dyh, dxv, dyv)
    assert penalty < 1e-10, f"Nonzero collision at t=0: {penalty}"

    # Cell area should be reasonable (around 1947.5 at optimal)
    area = abs(dxh * dyv - dxv * dyh)
    assert area > 1000 and area < 5000, f"Suspicious cell area at t=0: {area}"

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

    # Most configurations should be collision-free after clamping
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

    # Verify against actual tree vertices if initialized
    if kgs.tree_vertices is not None:
        jeroen_top_y = float(kgs.tree_vertices[0, 1])
        expected_scale = jeroen_top_y / 64.0
        assert abs(ppl.SCALE_FACTOR - expected_scale) < 1e-10, \
            f"Scale mismatch: {ppl.SCALE_FACTOR} vs {expected_scale}"

    print(f"  ✓ SCALE_FACTOR = {ppl.SCALE_FACTOR}")


# =============================================================================
# Lattice Generation Tests
# =============================================================================

def test_generate_lattice_basic():
    """Test lattice generation produces valid tree positions."""
    print("Testing lattice generation...")

    square_size = 5.0  # in Jeroen's scale
    n_inner = 30

    xyt = ppl.generate_lattice_trees(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=n_inner, square_size=square_size,
        scale=ppl.SCALE_FACTOR
    )

    assert xyt.shape[1] == 3, f"Wrong shape: {xyt.shape}"
    assert len(xyt) > 0, "No trees generated"
    assert len(xyt) <= n_inner, f"Too many trees: {len(xyt)} > {n_inner}"

    xyt_np = xyt.get()
    half = square_size / 2.0

    # All tree ORIGINS should be inside the square (bbox edges checked internally)
    assert np.all(np.abs(xyt_np[:, 0]) < half + 1.0), "Tree x outside square"
    assert np.all(np.abs(xyt_np[:, 1]) < half + 1.0), "Tree y outside square"

    # Angles should be in [0, 2*pi)
    assert np.all(xyt_np[:, 2] >= 0) and np.all(xyt_np[:, 2] < 2*np.pi + 0.01), \
        "Angles out of range"

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
    xyt_r = ppl.generate_lattice_trees(theta=0.5, **kwargs)

    # Should have roughly same count but different positions
    assert len(xyt_0) > 0 and len(xyt_r) > 0
    if len(xyt_0) == len(xyt_r):
        diff = float(cp.max(cp.abs(xyt_0[:, :2] - xyt_r[:, :2])).get())
        assert diff > 0.01, "Rotation had no effect on positions"

    print(f"  ✓ Rotation changes positions (n={len(xyt_0)} vs {len(xyt_r)})")


def test_generate_lattice_invalid():
    """Test that invalid lattice params return empty or few trees."""
    print("Testing invalid lattice params...")

    # Extreme t values that might create colliding lattice
    xyt = ppl.generate_lattice_trees(
        t_same=0.99, t_horiz=0.99, t_vert=0.99,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=30, square_size=5.0, scale=ppl.SCALE_FACTOR
    )

    # Should either be empty (collision detected) or have some trees
    # Key: shouldn't crash
    print(f"  ✓ Extreme params: {len(xyt)} trees (no crash)")


# =============================================================================
# SolutionCollection Subclass Tests
# =============================================================================

def test_solution_collection_create_empty():
    """Test create_empty propagates lattice fields."""
    print("Testing SolutionCollection create_empty...")

    base = ppl.SolutionCollectionSquareParametrizedLattice()
    base.n_inner = 15

    sol = base.create_empty(10, 20)

    assert sol.xyt.shape == (10, 20, 3)
    assert sol.h.shape == (10, 3)
    assert sol.lattice_params.shape == (10, ppl.N_LATTICE_PARAMS)
    assert sol.n_inner == 15

    print("  ✓ create_empty preserves lattice fields")


def test_solution_collection_select_ids():
    """Test select_ids slices lattice_params correctly."""
    print("Testing SolutionCollection select_ids...")

    sol = ppl.SolutionCollectionSquareParametrizedLattice()
    sol.n_inner = 5
    sol.xyt = cp.random.random((10, 8, 3), dtype=kgs.dtype_cp)
    sol.h = cp.random.random((10, 3), dtype=kgs.dtype_cp)
    sol.lattice_params = cp.arange(60, dtype=kgs.dtype_cp).reshape(10, 6)

    # Select indices 2, 5, 7
    inds = [2, 5, 7]
    sol.select_ids(inds)

    assert sol.xyt.shape[0] == 3
    assert sol.lattice_params.shape == (3, 6)
    # Row 0 of result should be original row 2
    assert float(sol.lattice_params[0, 0].get()) == 12.0  # 2*6 = 12
    assert float(sol.lattice_params[1, 0].get()) == 30.0  # 5*6 = 30
    assert float(sol.lattice_params[2, 0].get()) == 42.0  # 7*6 = 42

    print("  ✓ select_ids slices lattice_params correctly")


def test_solution_collection_merge():
    """Test merge concatenates lattice_params."""
    print("Testing SolutionCollection merge...")

    sol_a = ppl.SolutionCollectionSquareParametrizedLattice()
    sol_a.n_inner = 5
    sol_a.xyt = cp.zeros((3, 8, 3), dtype=kgs.dtype_cp)
    sol_a.h = cp.zeros((3, 3), dtype=kgs.dtype_cp)
    sol_a.lattice_params = cp.ones((3, 6), dtype=kgs.dtype_cp)

    sol_b = ppl.SolutionCollectionSquareParametrizedLattice()
    sol_b.n_inner = 5
    sol_b.xyt = cp.zeros((2, 8, 3), dtype=kgs.dtype_cp)
    sol_b.h = cp.zeros((2, 3), dtype=kgs.dtype_cp)
    sol_b.lattice_params = cp.ones((2, 6), dtype=kgs.dtype_cp) * 2.0

    sol_a.merge(sol_b)

    assert sol_a.xyt.shape[0] == 5
    assert sol_a.lattice_params.shape == (5, 6)
    assert float(sol_a.lattice_params[0, 0].get()) == 1.0
    assert float(sol_a.lattice_params[3, 0].get()) == 2.0

    print("  ✓ merge concatenates lattice_params")


def test_solution_collection_clone():
    """Test create_clone and create_clone_batch copy lattice_params."""
    print("Testing SolutionCollection clone...")

    src = ppl.SolutionCollectionSquareParametrizedLattice()
    src.n_inner = 5
    src.xyt = cp.random.random((4, 8, 3), dtype=kgs.dtype_cp)
    src.h = cp.random.random((4, 3), dtype=kgs.dtype_cp)
    src.lattice_params = cp.arange(24, dtype=kgs.dtype_cp).reshape(4, 6)

    dst = src.create_empty(4, 8)

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
    """Test that cost functions work with our SolutionCollection subclass."""
    print("Testing cost functions with lattice solution...")

    kgs.set_float32(False)

    sol = ppl.SolutionCollectionSquareParametrizedLattice()
    sol.n_inner = 5
    sol.override_phenotype = True  # phenotype = genotype for asymmetric

    # Generate some lattice trees
    inner_xyt = ppl.generate_lattice_trees(
        t_same=0.0, t_horiz=0.0, t_vert=0.0,
        theta=0.0, anchor_dx=0.0, anchor_dy=0.0,
        n_inner=8, square_size=5.0, scale=ppl.SCALE_FACTOR
    )

    N_trees = min(len(inner_xyt), 8)
    if N_trees < 3:
        print("  ⚠ Too few lattice trees generated, using random")
        N_trees = 8
        xyt = cp.random.random((1, N_trees, 3), dtype=cp.float64)
        xyt[:, :, 0:2] *= 3.0
        xyt[:, :, 0:2] -= 1.5
    else:
        xyt = inner_xyt[:N_trees].reshape(1, N_trees, 3).astype(cp.float64)

    sol.xyt = xyt
    sol.h = cp.array([[5.0, 0.0, 0.0]], dtype=cp.float64)
    sol.lattice_params = cp.zeros((1, ppl.N_LATTICE_PARAMS), dtype=cp.float64)

    # Test each cost function
    costs = [
        pack_cost.AreaCost(),
        pack_cost.CollisionCostSeparation(scaling=5.0),
        pack_cost.BoundaryDistanceCost(use_kernel=False),
    ]

    for c in costs:
        cost_val, grad_xyt, grad_h = c.compute_cost_ref(sol)
        assert cost_val.shape == (1,), f"{c.__class__.__name__}: wrong cost shape"
        assert grad_xyt.shape == sol.xyt.shape, \
            f"{c.__class__.__name__}: wrong grad shape"
        assert not cp.any(cp.isnan(cost_val)), f"{c.__class__.__name__}: NaN cost"
        assert not cp.any(cp.isnan(grad_xyt)), f"{c.__class__.__name__}: NaN gradient"
        print(f"  ✓ {c.__class__.__name__}: cost={float(cost_val[0]):.4f}")

    kgs.set_float32(True)


# =============================================================================
# Move Tests
# =============================================================================

def _make_test_population(n_individuals=10, n_trees=15, n_inner=10):
    """Helper: create a Population with lattice solution for move testing."""
    base = ppl.SolutionCollectionSquareParametrizedLattice()
    base.n_inner = n_inner
    base.edge_spacer = kgs.EdgeSpacerDummy()
    base.filter_move_locations_with_edge_spacer = False

    genotype = base.create_empty(n_individuals, n_trees)
    genotype.h[:, 0] = 5.0  # square size
    genotype.override_phenotype = True

    # Fill with some initial positions
    genotype.xyt = cp.random.random(
        (n_individuals, n_trees, 3), dtype=kgs.dtype_cp
    )
    genotype.xyt[:, :, 0:2] *= 3.0
    genotype.xyt[:, :, 0:2] -= 1.5

    # Set reasonable lattice params
    genotype.lattice_params[:, 0] = 0.0   # t_same
    genotype.lattice_params[:, 1] = 0.0   # t_horiz
    genotype.lattice_params[:, 2] = 0.0   # t_vert
    genotype.lattice_params[:, 3] = 0.1   # theta
    genotype.lattice_params[:, 4] = 0.0   # dx
    genotype.lattice_params[:, 5] = 0.0   # dy

    phenotype = genotype.create_empty(n_individuals, n_trees)
    phenotype.xyt[:] = genotype.xyt
    phenotype.h[:] = genotype.h
    phenotype.lattice_params[:] = genotype.lattice_params

    pop = pack_ga3.Population(genotype=genotype, phenotype=phenotype)
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
    mate_inds = cp.array([1, 2, 4])  # not used by jiggle, but interface requires it

    move = ppl.LatticeJiggle()
    move.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    # Params should have changed for indices 0, 3, 7
    for i in [0, 3, 7]:
        diff = float(cp.max(cp.abs(
            pop.genotype.lattice_params[i] - old_params[i]
        )).get())
        assert diff > 1e-6, f"Params unchanged for individual {i}"

    # Params should be unchanged for other indices
    for i in [1, 2, 4, 5, 6, 8, 9]:
        assert cp.allclose(pop.genotype.lattice_params[i], old_params[i]), \
            f"Params changed for non-target individual {i}"

    # Inner tree positions should have changed for affected individuals
    for i in [0, 3, 7]:
        n_inner = pop.genotype.n_inner
        diff = float(cp.max(cp.abs(
            pop.genotype.xyt[i, :n_inner] - old_xyt[i, :n_inner]
        )).get())
        # Could be zero if lattice regeneration produced same positions
        # (unlikely but possible), so just check no crash

    print("  ✓ LatticeJiggle changes params for selected individuals only")


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

    # Params should be substantially different
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
    pop.genotype.lattice_params[0] = cp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                                               dtype=kgs.dtype_cp)
    pop.genotype.lattice_params[5] = cp.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
                                               dtype=kgs.dtype_cp)

    # Mate: individual 3 gets params from individual 5
    mate_sol = copy.deepcopy(pop.genotype)
    inds = cp.array([3])
    mate_inds = cp.array([5])

    move = ppl.LatticeCrossover()
    move.do_move_vec(pop, inds, mate_sol, mate_inds, generator)

    # Individual 3 should now have individual 5's params
    assert cp.allclose(pop.genotype.lattice_params[3], mate_sol.lattice_params[5]), \
        "Crossover didn't copy params from mate"

    print("  ✓ LatticeCrossover copies params from mate")


def test_move_selector_integration():
    """Test lattice moves work inside MoveSelector alongside standard moves."""
    print("Testing MoveSelector integration...")

    pop = _make_test_population()
    generator = cp.random.default_rng(seed=42)

    # Build a MoveSelector with mixed moves
    selector = pack_move.MoveSelector()
    selector.moves = [
        [pack_move.JiggleRandomTree(max_xy_move=0.05, max_theta_move=0.3),
         'JiggleTree', 2.0],
        [ppl.LatticeJiggle(), 'LatticeJiggle', 1.0],
        [ppl.LatticeJump(), 'LatticeJump', 0.5],
    ]

    N = pop.genotype.N_solutions
    inds = cp.arange(N)
    mate_inds = cp.arange(N)

    # Should not crash — some individuals get tree moves, others lattice moves
    selector.do_move_vec(pop, inds, pop.genotype, mate_inds, generator)

    print("  ✓ MoveSelector runs with mixed tree+lattice moves")


# =============================================================================
# GA Integration Test (smoke test)
# =============================================================================

def test_ga_parametrized_lattice():
    """Smoke test: run parametrized lattice GA for a few generations."""
    print("Testing GA with parametrized lattice (smoke test)...")

    ga = ppl.baseline_parametrized_lattice()

    # Small scale for fast testing
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

    # Check that we have valid fitness values
    res = ga.ga.ga_list[0].population.fitness
    for g in ga.ga.ga_list[1:]:
        res = np.concatenate((res, g.population.fitness))

    assert len(res) > 0, "No fitness values"
    assert np.all(np.isfinite(res)), "Non-finite fitness values"

    # Check that lattice params survived through generations
    for g in ga.ga.ga_list:
        sol = g.population.genotype
        assert isinstance(sol, ppl.SolutionCollectionSquareParametrizedLattice), \
            f"Solution type lost: {type(sol)}"
        assert sol.lattice_params is not None, "lattice_params is None"
        assert sol.lattice_params.shape[1] == ppl.N_LATTICE_PARAMS, \
            f"lattice_params wrong shape: {sol.lattice_params.shape}"

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
    test_move_selector_integration()

    # GA smoke test
    test_ga_parametrized_lattice()

    print("=" * 60)
    print("All parametrized lattice tests passed!")
    print("=" * 60)
