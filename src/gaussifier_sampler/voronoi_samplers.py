"""GPU samplers for equal-mass Voronoi tessellations of a density map.

Two entry points, same algorithm:

* :func:`equal_mass_voronoi` — exact-N output, used for training and
  general inference. Calls into the fused C++/CUDA pipeline
  (``equal_mass_voronoi_native``) when the native extension is
  available; falls back to the step-by-step Python orchestrator
  (oversample → greedy merge → Lloyd) when not.

* :func:`equal_mass_voronoi_capturable` — fixed-size buffer + GPU-
  resident int32 count, suitable for ``torch.cuda.graph(...)``
  capture. Same count-aware Lloyd polish as the standard variant. CUDA-only.

Both share the algorithm:

    1. Oversample N · ``OVERSAMPLE_FACTOR`` points via the error-diffusion
       sampler.
    2. Remove zero-mass candidates, then merge low-mass cells into live
       Voronoi neighbors. Stalled parallel matches use a mass-preserving
       nearest-survivor fallback, so every round reaches its exact count.
    3. Polish with a few Lloyd iterations (density-weighted centroidal
       Voronoi) at the final point count.

The correctness contract:

* :func:`voronoi_assignment` is exact — every pixel is assigned to its
  nearest point. JFA gives a fast approximation, and we recover any
  point that JFA's pixel-snap initialization dropped via a focused exact
  cdist over the "lost" point subset.
* :func:`cell_masses` sums the density exactly into per-cell totals.
* :func:`equal_mass_voronoi` returns exactly ``n_points`` finite, in-bounds
  points without positional truncation.

All public functions accept points in normalized ``[0, 1]²`` coordinates
and densities of shape ``(H, W)``.
"""

from __future__ import annotations

import torch

from gaussifier_sampler import native_voronoi
from gaussifier_sampler.sampling import sample_density_points

# ----------------------------------------------------------------------- tunables

#: Initial point count is ``round(n_points · OVERSAMPLE_FACTOR)``. Higher values
#: yield better mass equalization (more candidates to cull) at the cost of more
#: work in the merge round. 1.5 is the empirical sweet spot.
OVERSAMPLE_FACTOR: float = 1.5

#: kNN search width in the merge round. Each candidate considers its k-1
#: nearest neighbors as merge targets in priority order.
DEFAULT_KNN_K: int = 8

#: Maximum greedy merge rounds. With OVERSAMPLE_FACTOR=1.5 and the per-round
#: cap of K/3 removals, only one round is needed in practice.
DEFAULT_MAX_MERGE_ROUNDS: int = 6

#: Per-round removal cap as a fraction of current K. With OVERSAMPLE_FACTOR=1.5,
#: 1/3 means a single merge round handles all removals. Switch to 1/6 for two
#: rounds at the cost of ~5 ms per call (each round runs voronoi_assignment +
#: adjacency); single round with mass-sorted Voronoi-adjacent target already
#: hits var/cell ~0.073 so the second round's marginal lift is small.
PER_ROUND_REMOVE_FRACTION: float = 1.0 / 3.0

#: Lloyd iterations after the final merge. Empirically 3 iters lands at
#: var/cell ~0.077 (vs ~0.083 with 1 iter) for ~1 ms extra. Diminishing
#: returns past 4.
FINAL_LLOYD_ITERS: int = 3

#: JFA propagates owner information across the grid in halving steps. We
#: start at twice the expected inter-point distance to skip wasted long-range
#: passes (mean inter-point distance ≈ √(H·W/N) pixels).
JFA_MIN_RADIUS_PX: int = 2


#: Chunk size for cdist computations, both for kNN fallback and the
#: voronoi-assignment lost-point recovery.
CDIST_CHUNK_PIXELS: int = 32_768

#: Bound memory in the rare nearest-survivor forced-merge fallback.
FORCED_MERGE_CDIST_CHUNK: int = 128

#: Cells below this mass do not contribute to the rendered result and are
#: removed before adjacency matching.
EMPTY_CELL_MASS_EPS: float = 1e-8

# ----------------------------------------------------------------------- internal


_PIXEL_GRID_CACHE: dict[tuple[int, int, str], torch.Tensor] = {}
_JFA_BUFFER_CACHE: dict[tuple[int, int, str], dict[str, torch.Tensor]] = {}
_JFA_COMPILED_CACHE: dict[tuple[int, int, tuple[int, ...]], object] = {}


def _compiled_jfa(height: int, width: int, schedule: tuple[int, ...]):
    """Return a torch.compile-d JFA-loop closure specialized for this
    (height, width, schedule) triple. Cached because torch.compile is
    expensive on first call."""
    key = (int(height), int(width), tuple(schedule))
    cached = _JFA_COMPILED_CACHE.get(key)
    if cached is not None:
        return cached

    def _runner(owner, cur_dist, points_pixel, pix_x, pix_y):
        return _jfa_passes(
            owner=owner,
            cur_dist=cur_dist,
            points_pixel=points_pixel,
            pix_x=pix_x,
            pix_y=pix_y,
            schedule=schedule,
            height=height,
            width=width,
        )

    # dynamic=True so we compile once per (height, width, schedule) and reuse
    # across all point counts; otherwise dynamo recompiles for each new N.
    try:
        compiled = torch.compile(_runner, dynamic=True, mode="reduce-overhead")
    except Exception:
        compiled = _runner
    _JFA_COMPILED_CACHE[key] = compiled
    return compiled


def _pixel_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Per-pixel center coordinates in normalized [0, 1]², row-major. Cached."""
    key = (int(height), int(width), str(device))
    cached = _PIXEL_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    grid = torch.stack([(xs + 0.5) / width, (ys + 0.5) / height], dim=-1).reshape(-1, 2)
    _PIXEL_GRID_CACHE[key] = grid
    return grid


def _jfa_buffers(height: int, width: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Persistent per-grid scratch tensors for voronoi_assignment.

    Avoids reallocating the (H, W) owner/distance tensors and the (H, W)
    pixel-coordinate maps on every call. Buffers are returned uninitialized;
    callers fill them.
    """
    key = (int(height), int(width), str(device))
    cached = _JFA_BUFFER_CACHE.get(key)
    if cached is not None:
        return cached
    # Scratch is mutated on every call. Allocate normal tensors even when the
    # first caller is inside inference_mode, so later eager calls may reuse it.
    with torch.inference_mode(False):
        ys, xs = torch.meshgrid(
            torch.arange(height, device=device, dtype=torch.float32),
            torch.arange(width, device=device, dtype=torch.float32),
            indexing="ij",
        )
        cached = {
            "pix_x": (xs + 0.5).contiguous(),
            "pix_y": (ys + 0.5).contiguous(),
            "owner": torch.empty((height, width), dtype=torch.long, device=device),
            "cur_dist": torch.empty((height, width), dtype=torch.float32, device=device),
        }
    _JFA_BUFFER_CACHE[key] = cached
    return cached


def _jfa_passes(
    *,
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
    pix_x: torch.Tensor,
    pix_y: torch.Tensor,
    schedule: tuple[int, ...],
    height: int,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The JFA inner loop. Pulled out so torch.compile can fuse the
    roll + mask + distance + where chain into one CUDA graph."""
    for s in schedule:
        for dy in (-s, 0, s):
            for dx in (-s, 0, s):
                if dx == 0 and dy == 0:
                    continue
                if abs(dy) >= height or abs(dx) >= width:
                    continue
                cand = torch.roll(owner, shifts=(-dy, -dx), dims=(0, 1))
                if dy > 0:
                    cand[height - dy :, :] = -1
                elif dy < 0:
                    cand[:-dy, :] = -1
                if dx > 0:
                    cand[:, width - dx :] = -1
                elif dx < 0:
                    cand[:, :-dx] = -1
                valid = cand >= 0
                own = cand.clamp(min=0)
                d = (pix_x - points_pixel[own, 0]).pow(2) + (pix_y - points_pixel[own, 1]).pow(2)
                update = valid & (d < cur_dist)
                owner = torch.where(update, cand, owner)
                cur_dist = torch.where(update, d, cur_dist)
    return owner, cur_dist


def _check_density(density: torch.Tensor) -> None:
    if density.ndim != 2:
        raise ValueError(f"density must be 2D, got shape {tuple(density.shape)}.")
    if density.shape[0] < 1 or density.shape[1] < 1:
        raise ValueError("density must have positive height and width.")


def _check_points(points: torch.Tensor) -> None:
    if points.ndim != 2 or points.shape[-1] != 2:
        raise ValueError(f"points must have shape [N, 2], got {tuple(points.shape)}.")


# ------------------------------------------------------------------- voronoi


def voronoi_assignment(
    points: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Assign each pixel of an HxW grid to its nearest point.

    Implementation: Jump Flooding (Rong & Tan 2006) with an O(log r)
    schedule sized to the expected inter-point distance, followed by a
    JFA+1 cleanup pass and a focused exact-cdist recovery for any points
    JFA's pixel-snap init lost. Result is exact for every input.

    Args:
        points: (N, 2) tensor of point coordinates in [0, 1]².
        height: pixel grid height.
        width: pixel grid width.

    Returns:
        (H·W,) int64 tensor of point indices in [0, N).
    """
    _check_points(points)
    if height < 1 or width < 1:
        raise ValueError(f"height/width must be ≥ 1, got ({height}, {width}).")
    device = points.device
    n = int(points.shape[0])
    if n == 0:
        raise ValueError("voronoi_assignment requires at least one point.")

    points_pixel = torch.stack(
        [
            (points[:, 0] * width).clamp(0.0, width - 1e-3),
            (points[:, 1] * height).clamp(0.0, height - 1e-3),
        ],
        dim=-1,
    ).to(torch.float32)

    buffers = _jfa_buffers(height, width, device)
    owner, cur_dist, pix_x, pix_y = (
        buffers["owner"],
        buffers["cur_dist"],
        buffers["pix_x"],
        buffers["pix_y"],
    )
    _seed_owner_map(owner, cur_dist, points_pixel, width)

    # Try native CUDA path first; fall back to PyTorch otherwise.
    use_native = native_voronoi.jfa_init_distance(owner, cur_dist, points_pixel)
    if not use_native:
        valid = owner >= 0
        own = owner.clamp(min=0)
        cur_dist = torch.where(
            valid,
            (pix_x - points_pixel[own, 0]).pow(2) + (pix_y - points_pixel[own, 1]).pow(2),
            cur_dist,
        )

    schedule = _jfa_schedule(height, width)
    if use_native:
        for s in schedule:
            native_voronoi.jfa_step(owner, cur_dist, points_pixel, s)
    elif device.type == "cuda":
        # torch.compile is only worthwhile on CUDA; the CPU codegen has had
        # bugs for this control-flow shape, and eager is competitive for tests.
        runner = _compiled_jfa(height, width, schedule)
        owner, cur_dist = runner(owner, cur_dist, points_pixel, pix_x, pix_y)
    else:
        owner, cur_dist = _jfa_passes(
            owner=owner,
            cur_dist=cur_dist,
            points_pixel=points_pixel,
            pix_x=pix_x,
            pix_y=pix_y,
            schedule=schedule,
            height=height,
            width=width,
        )

    flat_owner = _recover_lost_points(owner, cur_dist, points_pixel, pix_x, pix_y, use_native)
    # Detach + clone so callers receive a stable tensor that won't be
    # invalidated when voronoi_assignment is called again with the same H/W
    # (the underlying owner buffer is cached per-grid and mutated in-place).
    # Important when multiple sample passes share a renderer or when an
    # autograd graph holds the previous owner's values.
    return flat_owner.detach().clone()


def _seed_owner_map(
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
    width: int,
) -> None:
    """Reset the owner map to -1 and scatter every point into the pixel it lands on.

    When several points snap to the same pixel, the one closest to that pixel's center
    scatters last and wins; the others are repaired by the lost-point pass.
    """
    n = int(points_pixel.shape[0])
    owner.fill_(-1)
    cur_dist.fill_(float("inf"))
    px = points_pixel[:, 0].long()
    py = points_pixel[:, 1].long()
    point_indices = torch.arange(n, device=points_pixel.device, dtype=torch.long)
    flat_idx = py * width + px
    cell_offset_x = points_pixel[:, 0] - px.float() - 0.5
    cell_offset_y = points_pixel[:, 1] - py.float() - 0.5
    point_dist_to_cell = cell_offset_x.pow(2) + cell_offset_y.pow(2)
    insert_order = point_dist_to_cell.argsort(descending=True)
    owner.view(-1).scatter_(0, flat_idx[insert_order], point_indices[insert_order])


def _jfa_schedule(height: int, width: int) -> tuple[int, ...]:
    """Standard JFA steps: from the power of two at or above max(H, W) // 2 down to 1, then 1 again.

    Starting that high lets information reach every pixel even where no seed point lies
    within a few pixels (a shorter schedule can leave owner=-1 there); the final extra
    step-1 pass (JFA+1) cleans the common artifacts. The extra passes are cheap natively.
    """
    step = 1
    step_cap = max(1, max(height, width) // 2)
    while step < step_cap:
        step *= 2
    schedule: list[int] = []
    while step >= 1:
        schedule.append(step)
        step //= 2
    schedule.append(1)
    return tuple(schedule)


def _recover_lost_points(
    owner: torch.Tensor,
    cur_dist: torch.Tensor,
    points_pixel: torch.Tensor,
    pix_x: torch.Tensor,
    pix_y: torch.Tensor,
    use_native: bool,
) -> torch.Tensor:
    """Give territory back to points that JFA lost, and fill any pixel still unowned.

    A point is lost when its pixel-snap seed was overwritten and no pass propagated it. For
    each lost point an exact distance check runs against the current owners; if any pixel is
    still at -1, every point takes part so each pixel ends with a valid owner. Returns the
    flattened owner map.
    """
    n = int(points_pixel.shape[0])
    device = points_pixel.device
    flat_owner = owner.reshape(-1)
    flat_dist = cur_dist.reshape(-1)
    has_unfilled = bool((flat_owner < 0).any().item())
    seen = torch.zeros(n, dtype=torch.bool, device=device)
    seen[flat_owner.clamp(min=0)] = True
    lost = (~seen).nonzero(as_tuple=False).squeeze(-1)
    if has_unfilled:
        lost = torch.arange(n, device=device, dtype=torch.long)
    if lost.numel() == 0:
        return flat_owner
    if use_native and native_voronoi.lost_point_recovery(owner, cur_dist, points_pixel, lost):
        return flat_owner  # the native kernel updated owner and cur_dist in place
    lost_pix = points_pixel[lost]
    pixels = torch.stack([pix_x.reshape(-1), pix_y.reshape(-1)], dim=-1)
    n_pixels = pixels.shape[0]
    for start in range(0, n_pixels, CDIST_CHUNK_PIXELS):
        end = min(start + CDIST_CHUNK_PIXELS, n_pixels)
        block = pixels[start:end]
        d2 = (block.unsqueeze(1) - lost_pix.unsqueeze(0)).pow(2).sum(-1)
        min_d2, argmin = d2.min(dim=1)
        update = min_d2 < flat_dist[start:end]
        flat_owner[start:end] = torch.where(update, lost[argmin], flat_owner[start:end])
        flat_dist[start:end] = torch.where(update, min_d2, flat_dist[start:end])
    return flat_owner


def cell_masses(
    assignment: torch.Tensor,
    density: torch.Tensor,
    n_points: int,
) -> torch.Tensor:
    """Sum the density over each Voronoi cell. Exact — every pixel contributes."""
    _check_density(density)
    if assignment.numel() != density.numel():
        raise ValueError(
            f"assignment ({assignment.numel()}) and density ({density.numel()}) "
            "must have the same number of elements."
        )
    if n_points < 1:
        raise ValueError(f"n_points must be ≥ 1, got {n_points}.")
    flat = density.reshape(-1).to(torch.float32)
    masses = torch.zeros(n_points, dtype=torch.float32, device=density.device)
    masses.scatter_add_(0, assignment, flat)
    return masses


# ----------------------------------------------------------------------- knn


def lloyd_step(
    density: torch.Tensor,
    points: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """Density-weighted Lloyd iterations: each point becomes the mass-weighted
    centroid of its Voronoi cell. Empty cells stay put.

    Tries the native CUDA fused-accumulator path first (one launch for the
    three scatter-adds, one for finalize). Falls back to PyTorch otherwise.
    """
    _check_density(density)
    _check_points(points)
    if iterations <= 0:
        return points
    height, width = int(density.shape[0]), int(density.shape[1])
    device = density.device
    flat_density = density.reshape(-1).to(torch.float32)
    pixel_xy = _pixel_grid(height, width, device)

    use_native = device.type == "cuda"
    if use_native:
        # Mutable buffer; native kernel writes positions in place.
        pts = points.clone().contiguous().to(torch.float32)
        for _ in range(iterations):
            assignment = voronoi_assignment(pts, height, width)
            ok = native_voronoi.lloyd_step(pts, assignment, flat_density, pixel_xy)
            if not ok:
                use_native = False
                break
        if use_native:
            return pts

    # PyTorch fallback (for CPU / native build failure).
    weighted_x = flat_density * pixel_xy[:, 0]
    weighted_y = flat_density * pixel_xy[:, 1]
    for _ in range(iterations):
        n_points = points.shape[0]
        assignment = voronoi_assignment(points, height, width)
        wsum = torch.zeros(n_points, device=device, dtype=torch.float32)
        wsum.scatter_add_(0, assignment, flat_density)
        x_acc = torch.zeros(n_points, device=device, dtype=torch.float32)
        y_acc = torch.zeros(n_points, device=device, dtype=torch.float32)
        x_acc.scatter_add_(0, assignment, weighted_x)
        y_acc.scatter_add_(0, assignment, weighted_y)
        non_empty = wsum > 1e-8
        new_x = torch.where(non_empty, x_acc / wsum.clamp_min(1e-8), points[:, 0])
        new_y = torch.where(non_empty, y_acc / wsum.clamp_min(1e-8), points[:, 1])
        points = torch.stack([new_x, new_y], dim=-1).clamp(0.0, 1.0)
    return points


# ------------------------------------------------------------------- merge


def _voronoi_adjacency_neighbor(
    *,
    assignment: torch.Tensor,
    masses: torch.Tensor,
    height: int,
    width: int,
    knn_k: int = DEFAULT_KNN_K,
) -> torch.Tensor:
    """For every point, return its (n, k) Voronoi-adjacent neighbor list,
    sorted by ascending neighbor mass — lowest-mass adjacent cell first.

    Two cells are Voronoi-adjacent if they own a horizontally- or vertically-
    adjacent pair of pixels. Self is excluded. Padding uses the cell's own
    index so the caller's ``proposed != self`` check filters it out.

    Implementation: deduplicate each row's neighbor list on GPU using a
    stable lex sort over (src, dst), then keep only the lowest-mass entry
    per (src, dst) pair before re-sorting by mass. This avoids the per-row
    Python loop that dominates the merge round otherwise.
    """
    n = int(masses.shape[0])
    device = assignment.device
    a2d = assignment.view(height, width)

    # Collect all bidirectional adjacency pairs from h/v pixel neighbors.
    left = a2d[:, :-1].reshape(-1)
    right = a2d[:, 1:].reshape(-1)
    top = a2d[:-1, :].reshape(-1)
    bottom = a2d[1:, :].reshape(-1)
    src = torch.cat([left, right, top, bottom])
    dst = torch.cat([right, left, bottom, top])
    different = src != dst
    src = src[different]
    dst = dst[different]
    if src.numel() == 0:
        return torch.arange(n, device=device).view(n, 1).expand(n, knn_k).contiguous()

    # Deduplicate (src, dst) pairs: pack into one int64 key, take unique.
    pair_key = (src.to(torch.int64) * n + dst.to(torch.int64)).unique()
    src_u = pair_key // n
    dst_u = pair_key % n

    # Sort the unique adjacencies by (src, neighbor_mass) ascending. Stable
    # sort by mass first, then by src — yields lowest-mass-first within each src.
    dst_mass = masses[dst_u]
    by_mass = dst_mass.argsort()
    src_after_mass = src_u[by_mass]
    dst_after_mass = dst_u[by_mass]
    by_src = src_after_mass.argsort(stable=True)
    src_sorted = src_after_mass[by_src]
    dst_sorted = dst_after_mass[by_src]

    # Compute the rank of each adjacency within its src row (0 = lowest mass).
    counts = torch.bincount(src_sorted, minlength=n)
    starts = torch.zeros(n, dtype=torch.long, device=device)
    starts[1:] = torch.cumsum(counts[:-1], dim=0)
    positions = torch.arange(src_sorted.numel(), device=device)
    rank_in_src = positions - starts[src_sorted]
    keep = rank_in_src < knn_k
    src_kept = src_sorted[keep]
    dst_kept = dst_sorted[keep]
    rank_kept = rank_in_src[keep]

    # Scatter into the (n, knn_k) result. Initialize to self so any unfilled
    # slot fails the (proposed != self) check downstream.
    result = torch.arange(n, device=device).view(n, 1).expand(n, knn_k).contiguous()
    result[src_kept, rank_kept] = dst_kept
    return result


def _greedy_merge_round_impl(
    points: torch.Tensor,
    density: torch.Tensor,
    target_remove: int,
    knn_k: int = DEFAULT_KNN_K,
) -> torch.Tensor:
    """One pass of greedy merging via parallel-matching on GPU.

    The lowest-mass cells (up to ``target_remove`` of them) propose to merge
    into their Voronoi-adjacent alive neighbor with the lowest current mass.
    The Voronoi adjacency is extracted from the same assignment we already
    computed for cell masses, so it's free.

    Conflicts are resolved by lowest-priority-wins via
    ``scatter_reduce(amin)``: when two proposers want the same target, the
    one with the lowest current mass wins; the loser retries against its
    next-best Voronoi neighbor in the next column round.

    Total mass is preserved: every removed cell's mass is added to its
    merge target's mass via index_add.
    """
    _check_density(density)
    _check_points(points)
    if knn_k < 2:
        raise ValueError(f"knn_k must be ≥ 2 (need at least one non-self neighbor), got {knn_k}.")
    n = int(points.shape[0])
    if n <= 1 or target_remove <= 0:
        return points
    height, width = int(density.shape[0]), int(density.shape[1])

    assignment = voronoi_assignment(points, height, width)
    masses = cell_masses(assignment, density, n)
    targets_table = _voronoi_adjacency_neighbor(
        assignment=assignment,
        masses=masses,
        height=height,
        width=width,
        knn_k=knn_k,
    )
    if targets_table.shape[1] < 1:
        return points

    matched, target, alive = _match_low_mass_cells(targets_table, masses, target_remove)
    matched_idx = matched.nonzero(as_tuple=False).squeeze(-1)
    if matched_idx.numel() == 0:
        return points
    return _merge_matched_cells(points, masses, matched_idx, target[matched_idx])[alive]


def _match_low_mass_cells(
    targets_table: torch.Tensor, masses: torch.Tensor, target_remove: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel matching of the ``target_remove`` lowest-mass cells onto merge targets.

    Column ``c`` of ``targets_table`` is every cell's c-th lowest-mass Voronoi neighbour.
    Each round, unmatched candidates propose to their next column; when several propose to
    the same target, the lowest-mass proposer wins and the others retry in the next column.
    Returns ``(matched, target, alive)``: which cells merged away, into which cell, and the
    survivors.
    """
    n = int(masses.shape[0])
    device = masses.device
    sort_idx = masses.argsort()
    priority = torch.empty(n, dtype=torch.float32, device=device)
    priority[sort_idx] = torch.arange(n, dtype=torch.float32, device=device)
    is_candidate = priority < float(target_remove)

    alive = torch.ones(n, dtype=torch.bool, device=device)
    matched = torch.zeros(n, dtype=torch.bool, device=device)
    target = torch.full((n,), -1, dtype=torch.long, device=device)
    n_idx = torch.arange(n, device=device)
    INF = float("inf")

    for col in range(targets_table.shape[1]):
        proposers = is_candidate & alive & (~matched)
        proposed = targets_table[:, col]
        proposal_valid = proposers & alive[proposed] & (proposed != n_idx)
        my_priority = torch.where(
            proposal_valid,
            priority,
            torch.full_like(priority, INF),
        )
        target_min = torch.full((n,), INF, device=device, dtype=torch.float32)
        target_min.scatter_reduce_(0, proposed, my_priority, reduce="amin", include_self=False)
        wins = proposal_valid & (my_priority == target_min[proposed])
        # All updates below no-op when wins is empty (no GPU↔CPU sync needed).
        target = torch.where(wins, proposed, target)
        matched = matched | wins
        alive = alive & ~wins
    return matched, target, alive


def _merge_matched_cells(
    points: torch.Tensor, masses: torch.Tensor, matched_idx: torch.Tensor, target_idx: torch.Tensor
) -> torch.Tensor:
    """Move every merge target to the mass-weighted centroid of itself and its merged cells."""
    masses_post = masses.clone()
    weighted_pos = points * masses.unsqueeze(-1)
    weighted_pos.index_add_(0, target_idx, points[matched_idx] * masses[matched_idx].unsqueeze(-1))
    masses_post.index_add_(0, target_idx, masses[matched_idx])
    merged_points = weighted_pos / masses_post.unsqueeze(-1).clamp_min(EMPTY_CELL_MASS_EPS)
    return torch.where(
        (masses_post > EMPTY_CELL_MASS_EPS).unsqueeze(-1),
        merged_points,
        points,
    )


def _drop_empty_cells(
    points: torch.Tensor,
    masses: torch.Tensor,
    max_remove: int,
) -> tuple[torch.Tensor, int]:
    """Drop up to ``max_remove`` zero-mass points in stable index order."""
    if max_remove <= 0 or points.shape[0] <= 1:
        return points, 0
    empty = (masses <= EMPTY_CELL_MASS_EPS).nonzero(as_tuple=False).squeeze(-1)
    remove_count = min(int(empty.numel()), int(max_remove), int(points.shape[0]) - 1)
    if remove_count <= 0:
        return points, 0
    keep = torch.ones(points.shape[0], dtype=torch.bool, device=points.device)
    keep[empty[:remove_count]] = False
    return points[keep], remove_count


def _nearest_alive_targets(
    points: torch.Tensor,
    source_indices: torch.Tensor,
    alive: torch.Tensor,
    adjacency: torch.Tensor,
) -> torch.Tensor:
    """Resolve a live merge target for every forced-merge source."""
    source_adjacency = adjacency[source_indices]
    candidate_alive = alive[source_adjacency]
    has_adjacent_survivor = candidate_alive.any(dim=1)
    first_live_column = candidate_alive.to(torch.int64).argmax(dim=1)
    targets = source_adjacency.gather(1, first_live_column.unsqueeze(1)).squeeze(1)

    unresolved_rows = (~has_adjacent_survivor).nonzero(as_tuple=False).squeeze(-1)
    if unresolved_rows.numel() == 0:
        return targets

    survivor_indices = alive.nonzero(as_tuple=False).squeeze(-1)
    if survivor_indices.numel() == 0:
        raise RuntimeError("EMV forced merge has no surviving target point")
    survivor_points = points[survivor_indices]
    for start in range(0, int(unresolved_rows.numel()), FORCED_MERGE_CDIST_CHUNK):
        rows = unresolved_rows[start : start + FORCED_MERGE_CDIST_CHUNK]
        distances = torch.cdist(points[source_indices[rows]], survivor_points)
        targets[rows] = survivor_indices[distances.argmin(dim=1)]
    return targets


def _force_merge_round(
    points: torch.Tensor,
    density: torch.Tensor,
    target_remove: int,
    *,
    knn_k: int,
) -> torch.Tensor:
    """Complete stalled matching without deleting positive density mass."""
    n = int(points.shape[0])
    remove_count = min(max(0, int(target_remove)), max(0, n - 1))
    if remove_count <= 0:
        return points

    height, width = int(density.shape[0]), int(density.shape[1])
    assignment = voronoi_assignment(points, height, width)
    masses = cell_masses(assignment, density, n)
    adjacency = _voronoi_adjacency_neighbor(
        assignment=assignment,
        masses=masses,
        height=height,
        width=width,
        knn_k=knn_k,
    )
    source_indices = masses.argsort(stable=True)[:remove_count]
    alive = torch.ones(n, dtype=torch.bool, device=points.device)
    alive[source_indices] = False
    target_indices = _nearest_alive_targets(points, source_indices, alive, adjacency)

    masses_post = masses.clone()
    weighted_pos = points * masses.unsqueeze(-1)
    weighted_pos.index_add_(
        0,
        target_indices,
        points[source_indices] * masses[source_indices].unsqueeze(-1),
    )
    masses_post.index_add_(0, target_indices, masses[source_indices])
    merged_points = weighted_pos / masses_post.unsqueeze(-1).clamp_min(EMPTY_CELL_MASS_EPS)
    new_points = torch.where(
        (masses_post > EMPTY_CELL_MASS_EPS).unsqueeze(-1),
        merged_points,
        points,
    )
    return new_points[alive]


def greedy_merge_round(
    points: torch.Tensor,
    density: torch.Tensor,
    target_remove: int,
    knn_k: int = DEFAULT_KNN_K,
) -> torch.Tensor:
    """Remove exactly ``target_remove`` points while preserving cell mass."""
    _check_density(density)
    _check_points(points)
    if knn_k < 2:
        raise ValueError(f"knn_k must be >= 2, got {knn_k}.")
    n = int(points.shape[0])
    remove_count = min(max(0, int(target_remove)), max(0, n - 1))
    if remove_count <= 0:
        return points

    height, width = int(density.shape[0]), int(density.shape[1])
    assignment = voronoi_assignment(points, height, width)
    masses = cell_masses(assignment, density, n)
    remaining_points, empty_removed = _drop_empty_cells(points, masses, remove_count)
    remaining_remove = remove_count - empty_removed
    if remaining_remove <= 0:
        return remaining_points

    before_matching = int(remaining_points.shape[0])
    matched_points = _greedy_merge_round_impl(
        remaining_points,
        density,
        remaining_remove,
        knn_k,
    )
    matched_removed = before_matching - int(matched_points.shape[0])
    residual_remove = remaining_remove - matched_removed
    if residual_remove > 0:
        matched_points = _force_merge_round(
            matched_points,
            density,
            residual_remove,
            knn_k=knn_k,
        )

    expected = n - remove_count
    if int(matched_points.shape[0]) != expected:
        raise RuntimeError(
            "EMV merge violated exact-count contract: "
            f"expected {expected}, got {int(matched_points.shape[0])}"
        )
    return matched_points


# ----------------------------------------------------------- high-level API


def equal_mass_voronoi(
    density: torch.Tensor,
    n_points: int,
    *,
    seed: int = 0,
) -> torch.Tensor:
    """Sample exactly ``n_points`` points with approximately equal cell mass.

    Algorithm: error-diffusion oversample by ``OVERSAMPLE_FACTOR`` → greedy
    exact mass-preserving merge of low-mass cells → final Lloyd polish. Empty
    cells are removed before matching and positional truncation is forbidden.
    Returns ``(n_points, 2)`` in [0, 1]².

    Prefers the fully-fused native pipeline when the CUDA extension is loaded
    (one Python call into C++/CUDA; ~2 ms at N=10k, 512²). Falls back to the
    equivalent step-by-step Python orchestrator (with the same merge/polish
    hyperparameters) when the native module is unavailable.

    Expected output quality: var/cell ≈ 0.077 with no empty cells in the
    typical case. All hyperparameters are the module-level constants above.
    """
    _check_density(density)
    if n_points < 1:
        raise ValueError(f"n_points must be ≥ 1, got {n_points}.")

    native = _load_native_module()
    if native is not None and density.is_cuda and hasattr(native, "equal_mass_voronoi_native"):
        points = _native_equal_mass_voronoi(native, density, n_points, seed)
    else:
        points = _python_equal_mass_voronoi(density, n_points, seed)
    _check_placement(points, n_points)
    return points


def _native_equal_mass_voronoi(
    native, density: torch.Tensor, n_points: int, seed: int
) -> torch.Tensor:
    """The fused C++/CUDA pipeline: oversample, merge rounds and Lloyd polish in one call."""
    height, width = density.shape[-2], density.shape[-1]
    return native.equal_mass_voronoi_native(
        density.contiguous().float(),
        int(n_points),
        int(height),
        int(width),
        int(seed),
        float(OVERSAMPLE_FACTOR),
        int(DEFAULT_MAX_MERGE_ROUNDS),
        float(PER_ROUND_REMOVE_FRACTION),
        int(DEFAULT_KNN_K),
        int(FINAL_LLOYD_ITERS),
    )


def _python_equal_mass_voronoi(density: torch.Tensor, n_points: int, seed: int) -> torch.Tensor:
    """The same pipeline as step-by-step torch calls, with the same hyperparameters."""
    initial_count = max(n_points + 1, round(n_points * OVERSAMPLE_FACTOR))
    points = sample_density_points(density, initial_count, seed=seed)
    for _ in range(DEFAULT_MAX_MERGE_ROUNDS):
        current = int(points.shape[0])
        if current <= n_points:
            break
        excess = current - n_points
        per_round_cap = max(1, int(current * PER_ROUND_REMOVE_FRACTION))
        target_remove = min(excess, per_round_cap)
        points = greedy_merge_round(points, density, target_remove, knn_k=DEFAULT_KNN_K)
        if int(points.shape[0]) != current - target_remove:
            raise RuntimeError(
                "EMV placement made unexpected merge progress: "
                f"before={current}, requested={target_remove}, "
                f"after={int(points.shape[0])}"
            )
    if int(points.shape[0]) != int(n_points):
        raise RuntimeError(
            "EMV placement failed to reach the requested point count: "
            f"requested={int(n_points)}, actual={int(points.shape[0])}"
        )
    return lloyd_step(density, points, FINAL_LLOYD_ITERS)


def _check_placement(points: torch.Tensor, n_points: int) -> None:
    if points.shape != (int(n_points), 2):
        raise RuntimeError(
            "EMV placement failed to reach the requested point count: "
            f"expected {(int(n_points), 2)}, got {tuple(points.shape)}"
        )
    if not bool(torch.isfinite(points).all()):
        raise RuntimeError("EMV produced non-finite point coordinates")
    if bool(((points < 0.0) | (points > 1.0)).any()):
        raise RuntimeError("EMV produced point coordinates outside [0, 1]")


def equal_mass_voronoi_capturable(
    density: torch.Tensor,
    n_points: int,
    *,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-only equal-mass Voronoi with CUDA-Graph-friendly output.

    Returns ``(points_buf, count_buf)`` where ``points_buf`` is a fixed-size
    ``(round(OVERSAMPLE_FACTOR · n_points), 2)`` tensor and ``count_buf`` is a
    ``(1,)`` ``int32`` GPU-resident tensor holding the number of valid points.
    No host syncs along the pipeline; the whole call wraps cleanly in a
    ``torch.cuda.graph(...)`` capture.

    Quality matches the standard ``equal_mass_voronoi`` within noise — uses
    the same count-aware Lloyd polish under the hood.
    """
    _check_density(density)
    if n_points < 1:
        raise ValueError(f"n_points must be ≥ 1, got {n_points}.")
    native = _load_native_module()
    if native is None or not hasattr(native, "equal_mass_voronoi_native_capturable"):
        raise RuntimeError(
            "equal_mass_voronoi_capturable requires the CUDA extension; "
            "fall back to equal_mass_voronoi if running CPU-only."
        )
    height, width = density.shape[-2], density.shape[-1]
    return native.equal_mass_voronoi_native_capturable(
        density.contiguous().float(),
        int(n_points),
        int(height),
        int(width),
        int(seed),
        float(OVERSAMPLE_FACTOR),
        int(DEFAULT_MAX_MERGE_ROUNDS),
        float(PER_ROUND_REMOVE_FRACTION),
        int(DEFAULT_KNN_K),
        int(FINAL_LLOYD_ITERS),
    )


@torch.inference_mode()
def production_voronoi(
    density: torch.Tensor,
    n_points: int,
    *,
    seed: int = 12345,
) -> tuple[torch.Tensor, torch.Tensor]:
    """C++ production placement and fixed initial Voronoi-mass RGB weights.

    Pass the raw model density without anisotropy modulation or rescaling.
    Uses the capturable native EMV path and native owner assignment, exactly
    as the default production harness does. Returns exactly N owned points
    and N weights. The weights are computed once, before any XY correction.
    CUDA and the bundled native extension are required; there is no fallback
    to a different placement/assignment algorithm.
    """
    _check_density(density)
    if not density.is_cuda:
        raise ValueError("Production inference requires a CUDA image/density tensor.")
    native = _load_native_module()
    if native is None:
        raise RuntimeError("Production inference requires the native Voronoi CUDA extension.")
    raw_density = density.contiguous().float()
    buffer, count = equal_mass_voronoi_capturable(raw_density, n_points, seed=seed)
    actual = int(count.item())
    if actual != n_points:
        raise RuntimeError(f"Production EMV count mismatch: expected {n_points}, got {actual}.")
    # Own this slice so a future sampler call cannot reuse its backing storage.
    points = buffer[:actual].clone().contiguous()
    if points.shape != (n_points, 2) or not bool(torch.isfinite(points).all()):
        raise RuntimeError("Production EMV returned invalid point coordinates.")
    if bool(((points < 0) | (points > 1)).any()):
        raise RuntimeError("Production EMV returned coordinates outside [0, 1].")
    height, width = raw_density.shape
    assignment = native.voronoi_assignment_native(points, height, width)
    masses = cell_masses(assignment, raw_density, n_points)
    weights = masses / (masses.sum().clamp_min(1e-8) / n_points)
    if not bool(torch.isfinite(weights).all()):
        raise RuntimeError("Production EMV returned non-finite color weights.")
    return points, weights


def _load_native_module():
    """The CUDA extension, or None when CUDA is unavailable or it failed to build."""
    return native_voronoi.load_module()
