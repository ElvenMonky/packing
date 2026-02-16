"""Visualize parametrized lattice cells with tree outlines.

Uses same coordinate conventions as pack_vis_sol.

Usage from notebook:
    import viz_lattice
    params = [5.3797060e-01, -1.5120809e-01, -5.1772255e-01, 4.8861443e+01, 1.0111742e+01, -2.4611901e-02]
    viz_lattice.plot_lattice(*params, n_trees=50)
    
    # Or from param array:
    viz_lattice.plot_lattice_from_params(params, n_trees=50)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from shapely import affinity

import kaggle_support as kgs
import pack_parametrized_lattice as ppl
import pack_vis_sol


def plot_lattice(t_same, t_horiz, t_vert, theta_deg, anchor_dx=0.0, anchor_dy=0.0,
                 n_trees=50, square_size=5.0, p=2.0, shift_dx=0.0, shift_dy=0.0,
                 ax=None, show_square=True, show_cell_vectors=True):
    """Plot lattice trees using pack_vis_sol conventions.

    Args:
        t_same, t_horiz, t_vert: NFP parameters [-1, 1]
        theta_deg: rotation in degrees
        anchor_dx, anchor_dy: anchor offset (80x scale)
        n_trees: number of trees to generate
        square_size: square boundary (Jeroen scale)
        p: selection shape (1=square, >=sqrt(2)=circle)
        shift_dx, shift_dy: post-selection shift (80x scale)
        ax: matplotlib Axes (created if None)
        show_square: draw the square boundary
        show_cell_vectors: draw cell vectors from origin
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    scale = ppl.SCALE_FACTOR

    # Generate tree positions (numpy array, n_trees x 3)
    xyt_np = ppl.generate_lattice_trees(
        t_same, t_horiz, t_vert, theta_deg, anchor_dx, anchor_dy,
        n_trees, square_size, scale, p, shift_dx, shift_dy
    )

    # Build TreeList the same way pack_vis_sol does
    tree_list = kgs.TreeList()
    tree_list.xyt = xyt_np  # (N, 3) numpy — [x, y, angle_radians]
    thetas = xyt_np[:, 2]

    # Get Shapely tree polygons (handles rotation/translation internally)
    trees = tree_list.get_trees()

    # Draw bounding boxes and centers for each tree
    theta_rad = float(theta_deg) * np.pi / 180.0
    dx_same, dy_same, dxh, dyh, dxv, dyv = ppl.compute_cell_vectors(
        float(t_same), float(t_horiz), float(t_vert)
    )
    dxs_r, dys_r, dxh_r, dyh_r, dxv_r, dyv_r, bbx, bby, bbh, bbv = \
        ppl.rotate_and_align(dx_same, dy_same, dxh, dyh, dxv, dyv, theta_rad)

    bbh_j = bbh * scale  # bbox width in Jeroen coords
    bbv_j = bbv * scale  # bbox height in Jeroen coords
    bbx_j = bbx * scale
    bby_j = bby * scale

    for i in range(len(xyt_np)):
        tx, ty, angle = xyt_np[i]
        # Determine orientation
        is_down = abs(angle - (np.pi + theta_rad)) < 0.1
        orient = 1.0 if is_down else 0.0

        # bbox center = tree_origin - bbx*(2*orient-1), same for y
        bcx = tx - bbx_j * (2 * orient - 1)
        bcy = ty - bby_j * (2 * orient - 1)

        # Draw bbox rectangle (axis-aligned, centered on bbox center)
        hw, hh = bbh_j / 2, bbv_j / 2
        rect = MplPolygon(
            [(bcx - hw, bcy - hh), (bcx + hw, bcy - hh),
             (bcx + hw, bcy + hh), (bcx - hw, bcy + hh)],
            closed=True,
            facecolor='none',
            edgecolor='gray',
            linewidth=0.8,
            linestyle=':',
            zorder=1,
            alpha=0.6,
        )
        ax.add_patch(rect)

        # Mark bbox center
        color = '#F44336' if is_down else '#2196F3'
        ax.plot(bcx, bcy, 'x', color=color, markersize=5, zorder=4, alpha=0.7)

        # Mark tree origin
        ax.plot(tx, ty, '.', color=color, markersize=3, zorder=4, alpha=0.7)

    # Plot trees with rotation-based coloring (same as pack_vis_sol)
    pack_vis_sol._plot_polygons(trees, ax=ax, thetas=thetas, alpha=0.8)

    # Draw square boundary
    if show_square:
        half = square_size / 2.0
        square = Polygon([
            (-half, -half), (half, -half),
            (half, half), (-half, half)
        ])
        x, y = square.exterior.xy
        patch = MplPolygon(
            list(zip(x, y)),
            closed=True,
            facecolor='none',
            edgecolor='blue',
            linewidth=2.0,
            linestyle='--',
            zorder=2
        )
        ax.add_patch(patch)

    # Draw cell vectors (in Jeroen scale)
    if show_cell_vectors:
        # Scale vectors to Jeroen coords (reuse dxh_r etc from above)
        vec_h = np.array([dxh_r, dyh_r]) * scale
        vec_v = np.array([dxv_r, dyv_r]) * scale
        vec_s = np.array([dxs_r, dys_r]) * scale

        # Anchor = lattice position (0,0) shifted, which is a bbox center
        origin = np.array([anchor_dx * scale, anchor_dy * scale])

        ax.annotate('', xy=origin + vec_h, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.annotate('', xy=origin + vec_v, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
        ax.annotate('', xy=origin + vec_s, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color='orange', lw=2))

        ax.text(*(origin + vec_h * 1.05), 'h', color='red', fontsize=12, fontweight='bold')
        ax.text(*(origin + vec_v * 1.05), 'v', color='green', fontsize=12, fontweight='bold')
        ax.text(*(origin + vec_s * 1.05), 's', color='orange', fontsize=12, fontweight='bold')

        # Mark bbox center with dot
        ax.plot(*origin, 'ko', markersize=6, zorder=5)

    # Auto-scale from tree bounds
    all_bounds = [tree.bounds for tree in trees]
    if all_bounds:
        minx = min(b[0] for b in all_bounds)
        miny = min(b[1] for b in all_bounds)
        maxx = max(b[2] for b in all_bounds)
        maxy = max(b[3] for b in all_bounds)

        width = maxx - minx
        height = maxy - miny
        margin = max(width, height) * 0.05

        ax.set_xlim(minx - margin, maxx + margin)
        ax.set_ylim(miny - margin, maxy + margin)

    ax.set_aspect('equal', adjustable='box')
    ax.set_title(
        f't=({t_same:.3f}, {t_horiz:.3f}, {t_vert:.3f}) '
        f'θ={theta_deg:.1f}° anchor=({anchor_dx:.2f}, {anchor_dy:.2f})\n'
        f'{n_trees} trees, square={square_size:.2f}'
    )
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return ax


def plot_lattice_from_params(params, n_trees=None, square_size=5.0, **kwargs):
    """Convenience: plot from a 6, 8, or 10 element param array.

    Args:
        params: [t_same, t_horiz, t_vert, theta, anchor_dx, anchor_dy,
                 (n_inner), (p), (shift_dx), (shift_dy)]
        n_trees: override number of trees (default: params[6] or 50)
    """
    p = float(params[7]) if len(params) > 7 else 2.0
    shift_dx = float(params[8]) if len(params) > 8 else 0.0
    shift_dy = float(params[9]) if len(params) > 9 else 0.0
    if n_trees is None:
        n_trees = int(params[6]) if len(params) > 6 else 50
    return plot_lattice(
        params[0], params[1], params[2],
        params[3], params[4], params[5],
        n_trees=n_trees, square_size=square_size, p=p,
        shift_dx=shift_dx, shift_dy=shift_dy, **kwargs
    )
